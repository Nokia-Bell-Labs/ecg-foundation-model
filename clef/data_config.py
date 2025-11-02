from dataclasses import dataclass
from clef.data.mimiciv_dataloader import MIMICIVDataModule


data_mapping = {
    "mimiciv": MIMICIVDataModule,
}


@dataclass
class DataConfig:
    name: str

    def __post_init__(self):
        self.data_module_class = data_mapping[self.name]

