import importlib.util
from pathlib import Path

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "panviral"
    / "build_virus_inventory.py"
)

spec = importlib.util.spec_from_file_location("build_virus_inventory", MODULE_PATH)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_excludes_flu_and_covid_lineages():
    parent = {1: 1, 2: 1, 3: 2, 4: 3, 5: 4, 6: 1, 7: 6, 8: 7, 9: 8}
    name = {
        1: "Viruses",
        2: "Orthornavirae",
        3: "Orthomyxovirales",
        4: "Orthomyxoviridae",
        5: "Influenza A virus",
        6: "Riboviria",
        7: "Nidovirales",
        8: "Coronaviridae",
        9: "SARS-CoV-2",
    }

    assert mod.is_eukaryotic(5, parent, name) is False
    assert mod.is_eukaryotic(9, parent, name) is False


def test_keeps_other_eukaryotic_viruses():
    parent = {1: 1, 2: 1, 3: 2, 4: 3, 5: 4}
    name = {
        1: "Viruses",
        2: "Monodnaviria",
        3: "Baculoviridae",
        4: "Nucleopolyhedrovirus",
        5: "Autographa californica multiple nucleopolyhedrovirus",
    }

    assert mod.is_eukaryotic(5, parent, name) is True
