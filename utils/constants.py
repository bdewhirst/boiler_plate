from model import model


RAWCSV: str = "data/sundae-raw.csv"
SEED: int = 20220913
SAMPLE: int = 1000
DEP_VAR_COL_NAME: str = "y_yes"
# placeholder values which may not give great fit for current data
INDEP_VAR_COL_NAMES: str = [
    "age",
    "cons.price.idx",
]
SUPPORTED_MODEL_CLASSES: dict = {
    "global_naive": model.NaiveModel,
    #     "sm_linear": model.StatsModelsLinear,
    #     "sk_linear": model.SciKitLearnLinear,
    #     "logistic": model.SciKitLearnLogistic,
}
