from omegaconf import DictConfig

from src.model.feature_representations.aa_feature_model import AAFeaturizerModel


class OneHot(AAFeaturizerModel):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        print(f"Loading model: {self.cfg.embedder.model_name}")

    @property
    def aa_feature_mapping(self):
        # fmt: off
        aa_list = [
            'A', 'C', 'D', 'E', 'F',
            'G', 'H', 'I', 'K', 'L',
            'M', 'N', 'P', 'Q', 'R',
            'S', 'T', 'V', 'W', 'Y'
        ]
        # fmt: on

        aa_feature_mapping = {
            aa: [(1 if aa_list[i] == aa else 0) for i in range(len(aa_list))] for aa in aa_list
        }
        return aa_feature_mapping

    def validate_sequences(self, sequences: list) -> None:
        aa_alphabet = set(self.aa_feature_mapping.keys())
        for _, seq in enumerate(sequences):
            for char in seq:
                if char not in aa_alphabet:
                    raise ValueError(
                        "Sequence {i} contains a character {char} that is not in the alphabet: {aa_alphabet}"
                    )
