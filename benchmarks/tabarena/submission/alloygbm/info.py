from __future__ import annotations

from tabarena.models._method_metadata import MethodMetadata
from tabarena.models._model_info import ModelInfo
from tabarena.models.alloygbm.hpo import gen_alloygbm
from tabarena.models.alloygbm.model import AlloyGBMModel

# Updated to the measured deadline-support commit before the integration is run.
ALLOYGBM_REQUIREMENT = "alloygbm @ git+https://github.com/LGA-Personal/AlloyGBM.git@ecbafa1fb06130d5fe8e481e87d6aaa9fbb79a09"
alloygbm_method_metadata = MethodMetadata.config(
    method="AlloyGBM",
    ag_key="TA-ALLOYGBM",
    model_key="ALLOYGBM",
    display_name="AlloyGBM",
    compute="cpu",
    is_bag=True,
    can_hpo=True,
    config_default="TA-AlloyGBM_c1_default_BAG_L1",
    suite="tabarena-2026-10-20",
    date="2026-10-20",
    reference_url="https://github.com/LGA-Personal/AlloyGBM",
    license="MIT",
    commercial_use=True,
    verified=False,
)
alloygbm_info = ModelInfo(
    model_cls=AlloyGBMModel,
    search_space=gen_alloygbm,
    method_metadata=alloygbm_method_metadata,
    pip_extra=(ALLOYGBM_REQUIREMENT,),
)
