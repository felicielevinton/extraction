import numpy as np
from probes.Neuropixels import *


"""
Explications sur: https://billkarsh.github.io/SpikeGLX/help/imroTables/
"""


# todo: créer une fonction qui vérifie la validité des fichiers imro. Voir sur le github de SpikeGLX
def check_imro_is_valid():
    pass


def get_bank_and_channel(electrode_position, bank_length):
    """
    le test erreur se passera en amont.
    :param electrode_position:
    :param bank_length:
    :return:
    """
    _bank = electrode_position // (bank_length + 1)
    _channel = electrode_position - 1 - _bank * bank_length
    return _bank, _channel


def fill_pattern():
    pass


def imro(electrode_list, filename, reference=0, ap_band_gain=500, lfp_band_gain=250, hipass_ap_enabled=1, probe_type=0):
    """
    Generate imro file, imro stands for: IMec ReadOut file.
    (ChannelID Bank ReferenceIndex APGain LFPGain)
    :param filename: nom du fichier de sauvegarde .imro
    :param reference:
    :param electrode_list: liste des canaux à activer
    :param hipass_ap_enabled: on filtre avec passe-haut la bande des potentiels d'actions
    :param ap_band_gain: gain des potentiels d'action
    :param lfp_band_gain: gain des lfp
    :param probe_type: defaulted to 0 for Npx 1.0
    :return:
    """
    # TODO: que faire des références? => pour l'instant elles doivent être uniques
    # TODO: que faire des gains? => on vérifie que les gains sélectionnées sont dans la liste
    npx = Neuropixels1Point0()
    # TODO: verifier qu'il y a moins de 384 canaux
    # TODO: penser à supprimer les canaux de références {192,576,960}.
    n_electrodes = len(electrode_list)
    # ouvrir un fichier txt = .imro
    with open(filename, "w") as imro_file:
        # 1) on ajoute le type et le nombre de canaux
        list_channel_bank = list()
        for electrode_id in electrode_list:
            # vérifier si <= 960.
            list_channel_bank.append(get_bank_and_channel(electrode_id, npx.get_bank_size()))
        sorted_list = sorted(list_channel_bank, key=lambda bc: bc[1])  # on trie sur la deuxième position du tuple.
        __imro = f"({npx.get_probe_type()},{n_electrodes})"
        for tup in sorted_list:
            __imro += f"({tup[1]} {tup[0]} {reference} {ap_band_gain} {lfp_band_gain} {hipass_ap_enabled})"
        __imro += "\n"
        # todo: faire une fonction qui vérifie la validité du fichier .imro
        imro_file.write(__imro)


def create_tip_imro():
    desired_channels = np.arange(1, 385, dtype=np.int16)
    imro(electrode_list=desired_channels, filename="imro/tip.imro")


def create_middle_imro():
    desired_channels = np.arange(385, 769, dtype=np.int16)
    imro(electrode_list=desired_channels, filename="imro/middle.imro")


def create_end_imro():
    bank_2 = np.arange(769, 961, dtype=np.int16)
    chan_upper_bound = bank_2[-1] - 1 - 2 * 384
    fantom_bank = np.arange(chan_upper_bound + 1, 385)
    desired_channels = bank_2
    for elt in fantom_bank:
        desired_channels = np.append(desired_channels, [1 + elt + 1 * 384])
    imro(electrode_list=desired_channels, filename="imro/end.imro")


def create_span_all_probe_imro():
    """
    nope
    :return:
    """
    bank_0 = np.arange(1, 385, step=3)
    bank_1 = np.arange(385+1, 769, step=3)
    bank_2 = np.arange(769 + 2, 961, step=3)
    chan_upper_bound = bank_2[-1] - 1 - 2 * 384
    fantom_bank = np.arange(chan_upper_bound + 3, 385, step=3)
    for elt in range(len(fantom_bank)):
        if elt % 2 == 0:
            bank_0 = np.append(bank_0, [fantom_bank[elt]])
        else:
            bank_1 = np.append(bank_1, [1 + fantom_bank[elt] + 1 * 384])
    desired_channels = np.hstack((bank_0, bank_1, bank_2))
    imro(electrode_list=desired_channels, filename="imro/span.imro")


def create_tetrodes_imro():
    # 4 sélectionnés puis saute 4 ainsi de suite.
    pass


if __name__ == "__main__":
    create_span_all_probe_imro()
    create_tip_imro()
    create_middle_imro()
    create_end_imro()

