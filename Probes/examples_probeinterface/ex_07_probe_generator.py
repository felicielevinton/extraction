"""
Probe generator
---------------
`probeinterface` have also basic function to generate simple contact layouts like:
  * tetrodes
  * linear probes
  * multi-column probes
"""

##############################################################################
# Import

import numpy as np
import matplotlib.pyplot as plt

from probeinterface import Probe, ProbeGroup, io
from probeinterface.plotting import plot_probe, plot_probe_group

##############################################################################
# Generate 4 tetrodes:
#

from probeinterface import generate_tetrode

probegroup = ProbeGroup()
for i in range(4):
    tetrode = generate_tetrode()
    tetrode.move([i * 50, 0])
    probegroup.add_probe(tetrode)
probegroup.set_global_device_channel_indices(np.arange(16))

df = probegroup.to_dataframe()

plot_probe_group(probegroup, with_channel_index=True, same_axes=True)

##############################################################################
# Generate a linear probe:
#

from probeinterface import generate_linear_probe

linear_probe = generate_linear_probe(num_elec=16, ypitch=20)
plot_probe(linear_probe, with_channel_index=True)

##############################################################################
# Generate a multi-column probe:
#

from probeinterface import generate_multi_columns_probe

multi_columns = generate_multi_columns_probe(num_columns=3,
                                             num_contact_per_column=[10, 12, 10],
                                             xpitch=22, ypitch=20,
                                             y_shift_per_column=[0, -10, 0],
                                             contact_shapes='square', contact_shape_params={'width': 12})
plot_probe(multi_columns, with_channel_index=True, )

##############################################################################
# Generate a square probe:
#

square_probe = generate_multi_columns_probe(num_columns=12,
                                            num_contact_per_column=12,
                                            xpitch=10, ypitch=10,
                                            contact_shapes='square', contact_shape_params={'width': 8})
square_probe.create_auto_shape('rect')
plot_probe(square_probe)

fma = generate_multi_columns_probe(num_columns=4, num_contact_per_column=8,
                                   xpitch=40, ypitch=40, contact_shapes='square',
                                   contact_shape_params={'width': 32}, y_shift_per_column=[0, -40, 0, -40])


headstage = np.arange(32, dtype=int)
fma.create_auto_shape('rect')
fma.set_device_channel_indices(headstage)
group = probegroup
group.add_probe(fma)
io.write_prb("fma.prb", group)
plot_probe(fma, with_channel_index=True)
plt.show()
