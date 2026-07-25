from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from perception.contracts.base import ContractValidationError, JsonContract
from perception.contracts.capture import CaptureFrame
from perception.contracts.common import (
    CategoricalDistribution,
    NamedBelief,
    SCHEMA_VERSION,
)
from perception.contracts.relations import RelationPredicate
from perception.contracts.ssg import SSGSnapshot


@dataclass(frozen=True, slots=True)
class OntologyConcept(JsonContract):
    name: str
    description: str

    def __post_init__(self) -> None:
        if not self.name or not self.description:
            raise ContractValidationError(
                "ontology concepts require a name and description"
            )


@dataclass(frozen=True, slots=True)
class SimulatorClassMapping(JsonContract):
    exact_class: str
    semantic_class: str
    functional_types: tuple[str, ...]
    evidence: str

    def __post_init__(self) -> None:
        if not self.exact_class or not self.semantic_class or not self.evidence:
            raise ContractValidationError(
                "simulator mappings require class, semantic, and evidence"
            )
        if len(self.functional_types) != len(set(self.functional_types)):
            raise ContractValidationError("mapping functional types must be unique")


@dataclass(frozen=True, slots=True)
class SimulatorAlias(JsonContract):
    alias: str
    canonical_class: str

    def __post_init__(self) -> None:
        if not self.alias or not self.canonical_class:
            raise ContractValidationError("simulator aliases must be non-empty")


@dataclass(frozen=True, slots=True)
class Ontology(JsonContract):
    schema_version: str
    ontology_version: str
    semantic_classes: tuple[OntologyConcept, ...]
    functional_types: tuple[OntologyConcept, ...]
    simulator_mappings: tuple[SimulatorClassMapping, ...]
    simulator_aliases: tuple[SimulatorAlias, ...]
    relation_predicates: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ContractValidationError(
                f"unsupported schema_version {self.schema_version!r}"
            )
        if not self.ontology_version:
            raise ContractValidationError("ontology_version must be non-empty")
        semantic_names = _unique_names(self.semantic_classes, "semantic class")
        functional_names = _unique_names(self.functional_types, "functional type")
        exact_classes = [mapping.exact_class for mapping in self.simulator_mappings]
        if len(exact_classes) != len(set(exact_classes)):
            raise ContractValidationError("simulator exact classes must be unique")
        for mapping in self.simulator_mappings:
            if mapping.semantic_class not in semantic_names:
                raise ContractValidationError(
                    f"mapping {mapping.exact_class!r} uses unknown semantic class"
                )
            unknown_functions = set(mapping.functional_types) - functional_names
            if unknown_functions:
                raise ContractValidationError(
                    f"mapping {mapping.exact_class!r} uses unknown functional types "
                    f"{sorted(unknown_functions)}"
                )
        aliases = [alias.alias for alias in self.simulator_aliases]
        if len(aliases) != len(set(aliases)):
            raise ContractValidationError("simulator aliases must be unique")
        canonical = set(exact_classes)
        for alias in self.simulator_aliases:
            if alias.canonical_class not in canonical:
                raise ContractValidationError(
                    f"alias {alias.alias!r} points to unknown canonical class"
                )
        expected_relations = {predicate.value for predicate in RelationPredicate}
        if set(self.relation_predicates) != expected_relations:
            missing = expected_relations - set(self.relation_predicates)
            extra = set(self.relation_predicates) - expected_relations
            raise ContractValidationError(
                f"relation vocabulary mismatch; missing={sorted(missing)}, extra={sorted(extra)}"
            )
        if len(self.relation_predicates) != len(set(self.relation_predicates)):
            raise ContractValidationError("relation predicates must be unique")

    @property
    def semantic_names(self) -> frozenset[str]:
        return frozenset(item.name for item in self.semantic_classes)

    @property
    def functional_names(self) -> frozenset[str]:
        return frozenset(item.name for item in self.functional_types)

    def resolve_simulator_class(self, exact_class: str) -> str:
        alias_map = {
            item.alias: item.canonical_class for item in self.simulator_aliases
        }
        return alias_map.get(exact_class, exact_class)

    def mapping_for(self, exact_class: str) -> SimulatorClassMapping | None:
        canonical = self.resolve_simulator_class(exact_class)
        return next(
            (item for item in self.simulator_mappings if item.exact_class == canonical),
            None,
        )

    def validate_distribution(self, distribution: CategoricalDistribution) -> None:
        labels = {entry.label for entry in distribution.entries}
        unknown = labels - self.semantic_names
        if unknown:
            raise ContractValidationError(
                f"unknown semantic classes: {sorted(unknown)}"
            )

    def validate_functional_beliefs(self, beliefs: tuple[NamedBelief, ...]) -> None:
        unknown = {item.name for item in beliefs} - self.functional_names
        if unknown:
            raise ContractValidationError(
                f"unknown functional types: {sorted(unknown)}"
            )

    def validate_capture(self, frame: CaptureFrame) -> None:
        for item in frame.objects:
            if self.mapping_for(item.exact_simulator_class) is None:
                raise ContractValidationError(
                    f"unmapped simulator class {item.exact_simulator_class!r}"
                )
            self.validate_distribution(item.semantic_distribution)
            self.validate_functional_beliefs(item.functional_types)

    def validate_ssg(self, graph: SSGSnapshot) -> None:
        for node in graph.objects:
            self.validate_distribution(node.semantic_distribution)
            self.validate_functional_beliefs(node.functional_types)
            if (
                node.exact_simulator_class is not None
                and self.mapping_for(node.exact_simulator_class) is None
            ):
                raise ContractValidationError(
                    f"unmapped simulator class {node.exact_simulator_class!r}"
                )


def load_ontology(path: str | Path) -> Ontology:
    return Ontology.from_json_file(path)


def _unique_names(concepts: tuple[OntologyConcept, ...], label: str) -> set[str]:
    names = [item.name for item in concepts]
    if len(names) != len(set(names)):
        raise ContractValidationError(f"{label} names must be unique")
    return set(names)
