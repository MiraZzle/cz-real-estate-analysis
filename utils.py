import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def load_csv_to_dataframe(file_path: str) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.
    """
    df = pd.read_csv(file_path, encoding="utf-8")
    return df


def save_figure_to_pdf(fig, filename):
    """
    Save a figure to a PDF file.
    """
    with PdfPages(filename) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def save_figure_to_png(fig, filename):
    """
    Save a figure to a PNG file.
    """
    fig.savefig(filename, bbox_inches="tight", dpi=300)
    plt.close(fig)


KRAJE_NAMES = [
    "Hlavní město Praha",
    "Středočeský kraj",
    "Jihočeský kraj",
    "Plzeňský kraj",
    "Karlovarský kraj",
    "Ústecký kraj",
    "Liberecký kraj",
    "Královéhradecký kraj",
    "Pardubický kraj",
    "Vysočina",
    "Jihomoravský kraj",
    "Olomoucký kraj",
    "Zlínský kraj",
    "Moravskoslezský kraj",
]


KRAJ_DISTRICTS_RAW = """
    Hlavní město Praha
    Středočeský kraj
    Benešov
    Beroun
    Kladno
    Kolín
    Kutná Hora
    Mělník
    Mladá Boleslav
    Nymburk
    Praha-východ
    Praha-západ
    Příbram
    Rakovník
    Jihočeský kraj
    České Budějovice
    Český Krumlov
    Jindřichův Hradec
    Písek
    Prachatice
    Strakonice
    Tábor
    Plzeňský kraj
    Domažlice
    Klatovy
    Plzeň-jih
    Plzeň-město
    Plzeň-sever
    Rokycany
    Tachov
    Karlovarský kraj
    Cheb
    Karlovy Vary
    Sokolov
    Ústecký kraj
    Děčín
    Chomutov
    Litoměřice
    Louny
    Most
    Teplice
    Ústí nad Labem
    Liberecký kraj
    Česká Lípa
    Jablonec nad Nisou
    Liberec
    Semily
    Královéhradecký kraj
    Hradec Králové
    Jičín
    Náchod
    Rychnov nad Kněžnou
    Trutnov
    Pardubický kraj
    Chrudim
    Pardubice
    Svitavy
    Ústí nad Orlicí
    Kraj Vysočina
    Havlíčkův Brod
    Jihlava
    Pelhřimov
    Třebíč
    Žďár nad Sázavou
    Jihomoravský kraj
    Blansko
    Brno-město
    Brno-venkov
    Břeclav
    Hodonín
    Vyškov
    Znojmo
    Olomoucký kraj
    Jeseník
    Olomouc
    Prostějov
    Přerov
    Šumperk
    Zlínský kraj
    Kroměříž
    Uherské Hradiště
    Vsetín
    Zlín
    Moravskoslezský kraj
    Bruntál
    Frýdek-Místek
    Karviná
    Nový Jičín
    Opava
    Ostrava-město
    """
