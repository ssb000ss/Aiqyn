"""Auto-discovery registry for feature extractors."""

from __future__ import annotations

import importlib
import pkgutil
import structlog

from aiqyn.extractors.base import ExtractionContext, FeatureExtractor
from aiqyn.schemas import FeatureResult

log = structlog.get_logger(__name__)


class ExtractorRegistry:
    """Discovers and manages all FeatureExtractor implementations."""

    def __init__(self) -> None:
        self._extractors: dict[str, FeatureExtractor] = {}
        self._discovered = False

    def discover(self) -> None:
        """Auto-import all modules in aiqyn.extractors package."""
        if self._discovered:
            return

        import aiqyn.extractors as pkg

        for module_info in pkgutil.iter_modules(pkg.__path__):
            name = module_info.name
            if name in ("base", "registry"):
                continue
            try:
                module = importlib.import_module(f"aiqyn.extractors.{name}")
                # look for class named Extractor or any FeatureExtractor implementor
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (
                        isinstance(attr, type)
                        and attr_name not in ("FeatureExtractor",)
                        and isinstance(attr, type)
                    ):
                        try:
                            instance = attr()
                            if isinstance(instance, FeatureExtractor):
                                self._extractors[instance.feature_id] = instance
                                log.debug("extractor_registered", feature_id=instance.feature_id)
                        except Exception:
                            pass
            except Exception as exc:
                log.warning("extractor_import_failed", module=name, error=str(exc))

        self._discovered = True
        log.info("extractors_discovered", count=len(self._extractors))

    def get_enabled(self, enabled_ids: list[str]) -> list[FeatureExtractor]:
        self.discover()
        result = []
        for fid in enabled_ids:
            if fid in self._extractors:
                result.append(self._extractors[fid])
            else:
                log.warning("extractor_not_found", feature_id=fid)
        return result

    def get_all(self) -> list[FeatureExtractor]:
        self.discover()
        return list(self._extractors.values())

    @property
    def count(self) -> int:
        self.discover()
        return len(self._extractors)


_registry = ExtractorRegistry()


def get_registry() -> ExtractorRegistry:
    return _registry
