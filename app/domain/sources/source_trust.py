from __future__ import annotations

import csv
import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal, TypeAlias, cast


VerificationStatus: TypeAlias = Literal["active", "revoked"]


@dataclass(frozen=True, slots=True)
class VerifiedCompanyDomain:
    company: str
    domain: str
    evidence_reference: str
    verified_at: date
    verified_by: str
    status: VerificationStatus

    def __post_init__(self) -> None:
        required_values = {
            "company": self.company,
            "domain": self.domain,
            "evidence_reference": self.evidence_reference,
            "verified_by": self.verified_by,
        }
        for field_name, value in required_values.items():
            if not value.strip():
                raise ValueError(f"{field_name} must not be blank")

        if "://" in self.domain or "/" in self.domain:
            raise ValueError("domain must be a host name, not a URL")
        if self.status not in ("active", "revoked"):
            raise ValueError("status must be active or revoked")


class VerifiedDomainRegistry:
    def __init__(
        self,
        records: Iterable[VerifiedCompanyDomain] = (),
    ) -> None:
        self._records_by_company: dict[
            str,
            dict[str, VerifiedCompanyDomain],
        ] = {}
        for record in records:
            if record.status == "revoked":
                continue
            company_key = normalize_company(record.company)
            domain = normalize_domain(record.domain)
            company_domains = self._records_by_company.setdefault(
                company_key,
                {},
            )
            company_domains[domain] = record

    @classmethod
    def from_csv(cls, path: Path) -> VerifiedDomainRegistry:
        return cls(load_verified_domain_records(path))

    def find_matching_record(
        self,
        company: str,
        candidate_host: str,
    ) -> VerifiedCompanyDomain | None:
        company_key = normalize_company(company)
        host = normalize_domain(candidate_host)

        for verified_domain, record in self._records_by_company.get(
            company_key,
            {},
        ).items():
            if host == verified_domain or host.endswith(
                f".{verified_domain}"
            ):
                return record
        return None

    def find_matching_domain(
        self,
        company: str,
        candidate_host: str,
    ) -> str | None:
        record = self.find_matching_record(company, candidate_host)
        if record is None:
            return None
        return normalize_domain(record.domain)


def normalize_company(company: str) -> str:
    return company.strip().casefold()


def normalize_domain(domain: str) -> str:
    return domain.strip().lower().rstrip(".")


def verification_fingerprint(
    record: VerifiedCompanyDomain,
) -> str:
    fingerprint_source = "\x1f".join(
        [
            normalize_company(record.company),
            normalize_domain(record.domain),
            record.evidence_reference.strip(),
            record.verified_at.isoformat(),
            record.verified_by.strip().casefold(),
            record.status,
        ]
    )
    return hashlib.sha256(
        fingerprint_source.encode("utf-8")
    ).hexdigest()


REQUIRED_CSV_FIELDS = {
    "company",
    "domain",
    "evidence_reference",
    "verified_at",
    "verified_by",
    "status",
}


def load_verified_domain_records(
    path: Path,
) -> list[VerifiedCompanyDomain]:
    with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = set(reader.fieldnames or [])
        missing_fields = REQUIRED_CSV_FIELDS - fieldnames
        if missing_fields:
            missing = ", ".join(sorted(missing_fields))
            raise ValueError(
                f"verified domain CSV is missing fields: {missing}"
            )

        records: list[VerifiedCompanyDomain] = []
        for line_number, row in enumerate(reader, start=2):
            try:
                status_value = row["status"].strip().lower()
                if status_value not in ("active", "revoked"):
                    raise ValueError("status must be active or revoked")

                records.append(
                    VerifiedCompanyDomain(
                        company=row["company"],
                        domain=row["domain"],
                        evidence_reference=row["evidence_reference"],
                        verified_at=date.fromisoformat(
                            row["verified_at"].strip()
                        ),
                        verified_by=row["verified_by"],
                        status=cast(VerificationStatus, status_value),
                    )
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid verified domain CSV row {line_number}: "
                    f"{error}"
                ) from error

    return records
