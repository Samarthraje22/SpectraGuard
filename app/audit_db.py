"""Tamper-evident SHA-256 hash-chained audit ledger with SQLAlchemy + stdlib sqlite3 fallback."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any

try:
    from sqlalchemy import Float, Integer, String, create_engine, select
    from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

    SQLALCHEMY_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised on slim serverless runtimes (e.g. Vercel)
    SQLALCHEMY_AVAILABLE = False


GENESIS_HASH = "0" * 64


if SQLALCHEMY_AVAILABLE:

    class Base(DeclarativeBase):
        """SQLAlchemy declarative base."""

    class AuditRecord(Base):
        """Hash-chained provenance audit record."""

        __tablename__ = "audit_ledger"

        id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
        timestamp: Mapped[str] = mapped_column(String(64), nullable=False)
        audio_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
        verdict: Mapped[str] = mapped_column(String(32), nullable=False)
        spoof_prob: Mapped[float] = mapped_column(Float, nullable=False)
        watermark_hex: Mapped[str] = mapped_column(String(16), nullable=False)
        watermark_confidence: Mapped[float] = mapped_column(Float, nullable=False)
        inference_latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
        prev_hash: Mapped[str] = mapped_column(String(64), nullable=False)
        row_hash: Mapped[str] = mapped_column(String(64), nullable=False)

        def to_dict(self) -> dict[str, Any]:
            """Convert ORM row into a JSON-serializable dictionary."""
            return {
                "id": self.id,
                "timestamp": self.timestamp,
                "audio_sha256": self.audio_sha256,
                "verdict": self.verdict,
                "spoof_prob": self.spoof_prob,
                "watermark_hex": self.watermark_hex,
                "watermark_confidence": self.watermark_confidence,
                "inference_latency_ms": self.inference_latency_ms,
                "prev_hash": self.prev_hash,
                "row_hash": self.row_hash,
            }


def _compute_row_hash(payload: dict[str, Any]) -> str:
    """Compute deterministic SHA-256 over canonical JSON representation of row fields."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _sqlite_path_from_url(database_url: str) -> str:
    """Extract filesystem or :memory: path from a sqlite:/// URL."""
    if database_url.startswith("sqlite:////"):
        return "/" + database_url[len("sqlite:////") :]
    if database_url.startswith("sqlite:///"):
        return database_url[len("sqlite:///") :]
    return database_url


class ProvenanceLedger:
    """Thread-safe, tamper-evident SHA-256 hash-chained audit ledger."""

    def __init__(self, database_url: str | None = None, force_sqlite3: bool = False) -> None:
        default_db = (
            "sqlite:////tmp/spectraguard_audit.db"
            if os.getenv("VERCEL")
            else "sqlite:///./spectraguard_audit.db"
        )
        self.database_url = database_url or os.getenv("DATABASE_URL") or default_db
        self._lock = threading.Lock()
        self._use_sqlalchemy = SQLALCHEMY_AVAILABLE and not force_sqlite3

        if self._use_sqlalchemy:
            connect_args = (
                {"check_same_thread": False} if self.database_url.startswith("sqlite") else {}
            )
            self.engine = create_engine(self.database_url, connect_args=connect_args, future=True)
            Base.metadata.create_all(self.engine)
            self._session_factory = sessionmaker(
                bind=self.engine, expire_on_commit=False, class_=Session
            )
            self._conn = None
        else:
            db_path = _sqlite_path_from_url(self.database_url)
            self._conn = sqlite3.connect(db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_ledger (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    audio_sha256 TEXT NOT NULL,
                    verdict TEXT NOT NULL,
                    spoof_prob REAL NOT NULL,
                    watermark_hex TEXT NOT NULL,
                    watermark_confidence REAL NOT NULL,
                    inference_latency_ms REAL NOT NULL,
                    prev_hash TEXT NOT NULL,
                    row_hash TEXT NOT NULL
                )
                """
            )
            self._conn.commit()

    def append_event(
        self,
        *,
        audio_sha256: str,
        verdict: str,
        spoof_prob: float,
        watermark_hex: str,
        watermark_confidence: float,
        inference_latency_ms: float,
        timestamp: str | None = None,
    ) -> dict[str, Any]:
        """Append a hash-chained audit record and return its serialized dictionary."""
        ts = timestamp or datetime.now(timezone.utc).isoformat()
        rounded_spoof = round(float(spoof_prob), 6)
        rounded_wm_conf = round(float(watermark_confidence), 6)
        rounded_latency = round(float(inference_latency_ms), 4)

        with self._lock:
            if self._use_sqlalchemy:
                with self._session_factory() as session:
                    last_row = session.execute(
                        select(AuditRecord).order_by(AuditRecord.id.desc()).limit(1)
                    ).scalar_one_or_none()
                    prev_hash = last_row.row_hash if last_row is not None else GENESIS_HASH

                    hash_payload = {
                        "timestamp": ts,
                        "audio_sha256": audio_sha256,
                        "verdict": verdict,
                        "spoof_prob": rounded_spoof,
                        "watermark_hex": watermark_hex,
                        "watermark_confidence": rounded_wm_conf,
                        "inference_latency_ms": rounded_latency,
                        "prev_hash": prev_hash,
                    }
                    row_hash = _compute_row_hash(hash_payload)

                    record = AuditRecord(
                        timestamp=ts,
                        audio_sha256=audio_sha256,
                        verdict=verdict,
                        spoof_prob=rounded_spoof,
                        watermark_hex=watermark_hex,
                        watermark_confidence=rounded_wm_conf,
                        inference_latency_ms=rounded_latency,
                        prev_hash=prev_hash,
                        row_hash=row_hash,
                    )
                    session.add(record)
                    session.commit()
                    session.refresh(record)
                    return record.to_dict()

            assert self._conn is not None
            cur = self._conn.execute(
                "SELECT row_hash FROM audit_ledger ORDER BY id DESC LIMIT 1"
            ).fetchone()
            prev_hash = str(cur["row_hash"]) if cur is not None else GENESIS_HASH

            hash_payload = {
                "timestamp": ts,
                "audio_sha256": audio_sha256,
                "verdict": verdict,
                "spoof_prob": rounded_spoof,
                "watermark_hex": watermark_hex,
                "watermark_confidence": rounded_wm_conf,
                "inference_latency_ms": rounded_latency,
                "prev_hash": prev_hash,
            }
            row_hash = _compute_row_hash(hash_payload)
            insert_cur = self._conn.execute(
                """
                INSERT INTO audit_ledger (
                    timestamp, audio_sha256, verdict, spoof_prob,
                    watermark_hex, watermark_confidence, inference_latency_ms,
                    prev_hash, row_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts,
                    audio_sha256,
                    verdict,
                    rounded_spoof,
                    watermark_hex,
                    rounded_wm_conf,
                    rounded_latency,
                    prev_hash,
                    row_hash,
                ),
            )
            self._conn.commit()
            return {
                "id": int(insert_cur.lastrowid),
                "timestamp": ts,
                "audio_sha256": audio_sha256,
                "verdict": verdict,
                "spoof_prob": rounded_spoof,
                "watermark_hex": watermark_hex,
                "watermark_confidence": rounded_wm_conf,
                "inference_latency_ms": rounded_latency,
                "prev_hash": prev_hash,
                "row_hash": row_hash,
            }

    def verify_chain(self) -> dict[str, Any]:
        """Verify prev_hash linkage and SHA-256 integrity across all rows in ascending id order."""
        with self._lock:
            if self._use_sqlalchemy:
                with self._session_factory() as session:
                    rows = [
                        r.to_dict()
                        for r in session.execute(
                            select(AuditRecord).order_by(AuditRecord.id.asc())
                        )
                        .scalars()
                        .all()
                    ]
            else:
                assert self._conn is not None
                raw_rows = self._conn.execute(
                    "SELECT * FROM audit_ledger ORDER BY id ASC"
                ).fetchall()
                rows = [dict(r) for r in raw_rows]

        expected_prev = GENESIS_HASH
        for row in rows:
            if row["prev_hash"] != expected_prev:
                return {
                    "chain_intact": False,
                    "total_records": len(rows),
                    "corrupted_id": row["id"],
                }
            hash_payload = {
                "timestamp": row["timestamp"],
                "audio_sha256": row["audio_sha256"],
                "verdict": row["verdict"],
                "spoof_prob": round(float(row["spoof_prob"]), 6),
                "watermark_hex": row["watermark_hex"],
                "watermark_confidence": round(float(row["watermark_confidence"]), 6),
                "inference_latency_ms": round(float(row["inference_latency_ms"]), 4),
                "prev_hash": row["prev_hash"],
            }
            expected_hash = _compute_row_hash(hash_payload)
            if row["row_hash"] != expected_hash:
                return {
                    "chain_intact": False,
                    "total_records": len(rows),
                    "corrupted_id": row["id"],
                }
            expected_prev = row["row_hash"]

        return {
            "chain_intact": True,
            "total_records": len(rows),
            "corrupted_id": None,
        }

    def get_recent_logs(self, limit: int = 25) -> list[dict[str, Any]]:
        """Return the most recent audit records ordered newest-first."""
        with self._lock:
            if self._use_sqlalchemy:
                with self._session_factory() as session:
                    rows = session.execute(
                        select(AuditRecord).order_by(AuditRecord.id.desc()).limit(limit)
                    ).scalars().all()
                return [row.to_dict() for row in rows]

            assert self._conn is not None
            raw_rows = self._conn.execute(
                "SELECT * FROM audit_ledger ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in raw_rows]
