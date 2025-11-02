from dataclasses import dataclass
from typing import Optional

from clef.lightning.lightning_mimiciv import (
    LVEF1LeadDataModule,
    LVEF12LeadDataModule,
    LVEFClassifier,
    LVEFRegressor,
)
from clef.lightning.lightning_music import (
    MusicDataModule,
    MusicClassifier,
)
from clef.lightning.lightning_ptbxl import (
    PTBXL1LeadDataModule,
    PTBXL12LeadDataModule,
    PTBXLClassifier,
)
from clef.lightning.lightning_icentia import (
    BeatDataModule,
    RhythmDataModule,
    IcentiaClassifier,
)
from clef.lightning.lightning_chapman import (
    Chapman1LeadDataModule,
    Chapman12LeadDataModule,
    ChapmanClassifier,
)
from clef.lightning.lightning_aurorabp import (
    AuroraBPDataModule,
    AuroraBPRegressor,
)
from clef.lightning.lightning_mcmed import (
    MCMEDDataModule,
    MCMEDClassifier,
    MCMEDRegressor,
)

# Task components mapping
task_mapping = {
    "mimiciv": {
        "datamodule": {
            "1lead": LVEF1LeadDataModule,
            "12lead": LVEF12LeadDataModule,
        },
        "classifier": LVEFClassifier,
        "regressor": LVEFRegressor,
    },
    "music": {
        "datamodule": MusicDataModule,
        "classifier": MusicClassifier,
        "regressor": None,  # MUSIC dataset only has classification
    },
    "ptbxl": {
        "datamodule": {
            "1lead": PTBXL1LeadDataModule,
            "12lead": PTBXL12LeadDataModule,
        },
        "classifier": PTBXLClassifier,
        "regressor": None,  # PTB-XL dataset only has classification
    },
    "icentia": {
        "datamodule": {
            "beat": BeatDataModule,
            "rhythm": RhythmDataModule,
        },
        "classifier": IcentiaClassifier,
        "regressor": None,  # Icentia dataset only has classification
    },
    "chapman": {
        "datamodule": {
            "1lead": Chapman1LeadDataModule,
            "12lead": Chapman12LeadDataModule,
        },
        "classifier": ChapmanClassifier,
        "regressor": None,  # Chapman dataset only has classification
    },
    "aurorabp": {
        "datamodule": AuroraBPDataModule,
        "classifier": None,  # AuroraBP dataset only has regression
        "regressor": AuroraBPRegressor,  
    },
    "mcmed": {
        "datamodule": MCMEDDataModule,
        "classifier": MCMEDClassifier,
        "regressor": MCMEDRegressor,
    },
}

ptbxl_num_classes = {
    "all": 71,
    "diagnostic": 44,
    "subdiagnostic": 23,
    "superdiagnostic": 5,
    "form": 19,
    "rhythm": 12,
}

mcmed_num_classes = {
    "ed-dispo": 4,
    "dc-dispo": 6,
    "acuity": 5,
}

icentia_num_classes = {
    "beat": 3,  # N, S, V (Q is not considered a class here)
    "rhythm": 3,  # N, AFIB, AFL
}

@dataclass
class TaskConfig:
    name: str
    task: Optional[str] = None
    lead_config: Optional[str] = None  # "1lead" or "12lead" for mimiciv/ptbxl

    def __post_init__(self):
        if self.name not in task_mapping:
            raise ValueError(
                f"Unknown task: {self.name}. Available tasks: {list(task_mapping.keys())}"
            )

        datamodule_entry = task_mapping[self.name]["datamodule"]
        if isinstance(datamodule_entry, dict):
            if self.lead_config in datamodule_entry:
                # 12lead or 1lead
                self.datamodule_class = datamodule_entry[self.lead_config]
            elif self.task in datamodule_entry:
                # beat or rhythm for icentia
                self.datamodule_class = datamodule_entry[self.task]
            else:
                raise ValueError("Please specify a valid lead_config or task_focus")
        else:
            self.datamodule_class = datamodule_entry

        self.classifier_class = task_mapping[self.name]["classifier"]
        self.regressor_class = task_mapping[self.name]["regressor"]

    def get_model_class(self, task_type):
        """Get the appropriate model class based on task type."""
        if task_type == "classification":
            if self.classifier_class is None:
                raise ValueError(f"Task {self.name} does not support classification")
            return self.classifier_class
        elif task_type == "regression":
            if self.regressor_class is None:
                raise ValueError(f"Task {self.name} does not support regression")
            return self.regressor_class
        else:
            raise ValueError(f"Unknown task type: {task_type}")

    def get_ptbxl_num_classes(self):
        """Get the number of classes for a specific PTB-XL task."""
        if self.name != "ptbxl":
            return None

        if self.task not in ptbxl_num_classes:
            raise ValueError(
                f"Unknown PTB-XL task: {self.task}. Available tasks: {list(ptbxl_num_classes.keys())}"
            )

        return ptbxl_num_classes[self.task]
    
    def get_mcmed_num_classes(self):
        """Get the number of classes for a specific MCMED task."""
        if self.name != "mcmed":
            return None

        if self.task not in mcmed_num_classes:
            raise ValueError(
                f"Unknown MCMED task: {self.task}. Available tasks: {list(mcmed_num_classes.keys())}"
            )

        return mcmed_num_classes[self.task]
