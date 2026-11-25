"""Tamper-evident SHA-256 hash-chained SQLAlchemy audit ledger."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import Float, Integer, String, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


GENESIS_HASH = "0" * 64


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


class ProvenanceLedger:
    """Thread-safe, tamper-evident SHA-256 hash-chained audit ledger."""

    def __init__(self, database_url: str | None = None) -> None:
        default_db = (
            "sqlite:////tmp/spectraguard_audit.db"
            if os.getenv("VERCEL")
            else "sqlite:///./spectraguard_audit.db"
        )
        self.database_url = database_url or os.getenv("DATABASE_URL") or default_db
        connect_args = {"check_same_thread": False} if self.database_url.startswith("sqlite") else {}
        self.engine = create_engine(self.database_url, connect_args=connect_args, future=True)
        Base.metadata.create_all(self.engine)
        self._session_factory = sessionmaker(bind=self.engine, expire_on_commit=False, class_=Session)
        self._lock = threading.Lock()

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

    def verify_chain(self) -> dict[str, Any]:
        """Verify prev_hash linkage and SHA-256 integrity across all rows in ascending id order."""
        with self._lock:
            with self._session_factory() as session:
                rows = session.execute(
                    select(AuditRecord).order_by(AuditRecord.id.asc())
                ).scalars().all()

        expected_prev = GENESIS_HASH
        for row in rows:
            if row.prev_hash != expected_prev:
                return {
                    "chain_intact": False,
                    "total_records": len(rows),
                    "corrupted_id": row.id,
                }
            hash_payload = {
                "timestamp": row.timestamp,
                "audio_sha256": row.audio_sha256,
                "verdict": row.verdict,
                "spoof_prob": round(float(row.spoof_prob), 6),
                "watermark_hex": row.watermark_hex,
                "watermark_confidence": round(float(row.watermark_confidence), 6),
                "inference_latency_ms": round(float(row.inference_latency_ms), 4),
                "prev_hash": row.prev_hash,
            }
            expected_hash = _compute_row_hash(hash_payload)
            if row.row_hash != expected_hash:
                return {
                    "chain_intact": False,
                    "total_records": len(rows),
                    "corrupted_id": row.id,
                }
            expected_prev = row.row_hash

        return {
            "chain_intact": True,
            "total_records": len(rows),
            "corrupted_id": None,
        }

    def get_recent_logs(self, limit: int = 25) -> list[dict[str, Any]]:
        """Return the most recent audit records ordered newest-first."""
        with self._lock:
            with self._session_factory() as session:
                rows = session.execute(
                    select(AuditRecord).order_by(AuditRecord.id.desc()).limit(limit)
                ).scalars().all()
        return [row.to_dict() for row in rows]
