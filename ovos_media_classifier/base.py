from abc import ABC, abstractmethod
from typing import Tuple, List, Optional

from ovos_media_classifier.intents import MediaType, OCPDomain


class AbstractMediaClassifier(ABC):
    """Classifies a natural-language query into an OCP MediaType and domain.

    Implementors must provide:
      classify()        — (MediaType, confidence) for the ocp_play domain
      classify_domain() — (OCPDomain, confidence); default derives from classify()
      is_ocp_query()    — (bool, confidence); default derives from classify_domain()

    Subclasses that have a cheap domain head (e.g. M2V, padatious with a
    separate domain container) should override classify_domain() directly.
    Subclasses that also handle ocp_control (padatious) may override
    is_ocp_query() to return True for control intents too.
    """

    @abstractmethod
    def classify(
        self,
        query: str,
        lang: str,
        valid_labels: Optional[List[MediaType]] = None,
    ) -> Tuple[MediaType, float]:
        """Return the most likely MediaType and a confidence in [0, 1].

        Args:
            query: User utterance.
            lang:  BCP-47 language tag (should already be standardised).
            valid_labels: When provided, only return one of these types.
                          Return (GENERIC, 0.0) when nothing matches.
        """

    def classify_domain(self, query: str, lang: str) -> Tuple[OCPDomain, float]:
        """Classify the top-level OCP domain: ocp_play / ocp_control / not_ocp.

        The default implementation delegates to classify() — if a non-GENERIC
        MediaType is returned it infers OCP_PLAY; otherwise NOT_OCP.

        Subclasses with a dedicated domain head (M2V, padatious domain
        container, sklearn domain model) should override this for better
        accuracy, especially to detect OCP_CONTROL intents.
        """
        media_type, conf = self.classify(query, lang)
        if media_type != MediaType.GENERIC:
            return OCPDomain.OCP_PLAY, conf
        return OCPDomain.NOT_OCP, 0.0

    def is_ocp_query(self, query: str, lang: str) -> Tuple[bool, float]:
        """Return (True, conf) if the query targets OCP (play or control).

        The default implementation delegates to classify_domain().
        """
        domain, conf = self.classify_domain(query, lang)
        return domain != OCPDomain.NOT_OCP, conf
