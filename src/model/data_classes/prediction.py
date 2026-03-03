from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Prediction:
    """Dataclass for organizing the output and intermediate values from the model"""

    sequences: np.ndarray
    sequence_embeddings: np.ndarray
    additional_features_untransformed: pd.DataFrame
    features: np.ndarray
    features_concat_dimred_concat: np.ndarray | None
    predictions: np.ma.masked_array

    # TODO: convert to masked array?
    predictions_scaled: np.ndarray
    data_split: str

    # Labels only available during training
    labels: np.ndarray = None
    group_names: list[str] = None

    # Not all methods provide a probability
    predictions_probability: np.ma.masked_array = None
    residue_level_prediction: bool = False

    def get_dataframe(self):
        if self.sequences.shape[0] == 0:
            return pd.DataFrame()

        # create the sequence-level table
        df = pd.DataFrame(
            {
                "sequence": self.sequences,
                "data_split": self.data_split,
                "group": self.group_names if self.group_names is not None else "",
            }
        )
        df = (
            pd.concat(
                [self.additional_features_untransformed.reset_index(drop=True), df],
                axis=1,
            )
            if self.additional_features_untransformed is not None
            else df
        )

        # expand to residue-level table, if applicable
        if self.residue_level_prediction:
            df["sequence_length_aa"] = df["sequence"].apply(len)
            explode_length = df["sequence_length_aa"].sum()
            df["res_aa"] = df["sequence"].apply(list)
            df["res_aa_idx"] = df["sequence"].apply(lambda x: range(len(x)))
            df["res_value_bool"] = (
                [[int(y) for y in list(x)] for x in self.labels]
                if self.labels is not None
                else None
            )
            df = df.explode(["res_aa", "res_aa_idx", "res_value_bool"]).reset_index(
                names="sequence_idx"
            )
            if len(df) != explode_length:
                raise ValueError(
                    f"Mismatched number of rows in residue table. Expected {explode_length} and found {len(df)}"
                )

            # add the labels and predictions
            df["y"] = np.concatenate(
                [np.array([float(y) for y in list(x)], dtype=float) for x in self.labels]
            )
            df["y_pred"] = self.predictions.compressed()
            df["y_pred_prob"] = self.predictions_probability.compressed()

        else:
            # all the labels and predictions are already flattened
            df["y"] = self.labels
            df["y_pred"] = self.predictions
            df["y_pred_prob"] = (
                self.predictions_probability if self.predictions_probability is not None else np.nan
            )

            if self.additional_features_untransformed is not None:
                df = pd.concat(
                    [self.additional_features_untransformed.reset_index(drop=True), df],
                    axis=1,
                )

        return df

    def describe_data(self):
        for name, val in vars(self).items():
            if not name.startswith("__") and val is not None:
                if val is not None and type(val) not in [str, bool]:
                    shape = getattr(val, "shape", None)
                    shape = shape if shape else len(val)
                    print(f"{name}: {shape}")
                else:
                    print(f"{name}: {val}")
