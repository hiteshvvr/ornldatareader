import h5py as hd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl

filename = "./July/FNPB_1027.nxs.h5"
afile = hd.File(filename)
data = afile['entry']['bank1_events']['event_id'][()]
timeoff = afile['entry']['bank1_events']['event_time_offset'][()]
timezer = afile['entry']['bank1_events']['event_time_zero'][()]

# data1 = afile['entry']['bank_unmapped_events']['event_id'][()]
# timeoff1 = afile['entry']['bank_unmapped_events']['event_time_offset'][()]
# timezer1 = afile['entry']['bank_unmapped_events']['event_time_zero'][()]

# Tube numbering start from bottom, 
# Pixel number start from beam left

duration = afile['entry']['duration'][()][0]
duration
proton_charge = afile['entry']['proton_charge'][()][0]
proton_charge

# data = np.append(data, data1)
# timeoff = np.append(timeoff, timeoff1)
# timezer = np.append(timezer, timezer1)
print(len(data), len(timeoff), len(timezer))


df = pd.DataFrame({'pixel':data, 'timeoffset':timeoff})#, 'timezer':timezer})
df['pix'] =  df.pixel%256
df['pix'] =  df['pix']*400/255

# df = df.query("pix > 70")

bins = 256*np.arange(9)-1
tube = np.array([1,2,3,4,5,6,7,8])
cat = pd.cut(df.pixel, bins = bins, labels=tube).to_numpy()
df['tube'] = cat


white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
    (0, '#ffffff'),
    (1e-20, '#440053'),
    (0.2, '#404388'),
    (0.4, '#2a788e'),
    (0.6, '#21a784'),
    (0.8, '#78d151'),
    (1, '#fde624'),
], N=512)


fig, ax = plt.subplots(nrows=2, ncols=2)
fig.suptitle(filename, fontsize=14, fontweight='bold')
# df.pivot(columns='tube').pix.plot.hist(bins = np.arange(0,256), histtype = 'step', ax = ax[0,0])
# df.pivot(columns='tube').pix.plot.hist(bins = np.arange(0,255), histtype = 'step', ax = ax[0,0], logy =False)
# ax[0,0].set_xlabel('bins or pixels', fontsize=10)
df.pivot(columns='tube').pix.plot.hist(bins = np.linspace(0,402, 256), histtype = 'step', ax = ax[0,0], logy =False, weights = np.ones(len(df.timeoffset.to_numpy())) / duration)
ax[0,0].set_xlabel('pixels(mm)', fontsize=10)
ax[0,0].set_ylabel('count rate for run', fontsize=10)
# df.pivot(columns='tube').timeoffset.plot.hist(bins = 1000, histtype = 'step', ax = ax[0,1])
df.timeoffset.plot.hist(bins = 1000, histtype = 'step', ax = ax[0,1], weights = np.ones(len(df.timeoffset.to_numpy())) / duration)
ax[0,1].set_xlabel('Time of flight(us)', fontsize=10)
ax[0,1].set_ylabel('count rate for run', fontsize=10)
# a = ax[1,0].hist2d(df.pix, df.tube, norm=mpl.colors.LogNorm(), bins=(np.arange(0,256),np.arange(0,9)), cmap=white_viridis)
# a = ax[1,0].hist2d(df.pix, df.tube, bins=(np.arange(0,256),np.arange(0,9)), cmap=white_viridis)
# ax[1,0].set_xlabel('bins or pixels', fontsize=10)
a = ax[1,0].hist2d(df.pix, df.tube, bins=(np.linspace(0,402, 256),np.arange(0,9)), cmap='viridis', weights = np.ones(len(df.timeoffset.to_numpy())) / duration)
ax[1,0].text(0,0, "bottom")
ax[1,0].text(350, 4, "Beam\nleft")
ax[1,0].set_xlabel('pixels(mm)', fontsize=10)
ax[1,0].set_ylabel('tube', fontsize=10)
# plt.colorbar()
cb1=plt.colorbar(a[3], ax=ax[1,0])#, shrink=0.80)
# df.timeoffset.hvplot.hist(bins = 300)


ax[1,1].axis([0, 10, 0, 10])
print(df.pix.shape)
print(duration)
ax[1,1].text(3, 8, 'Count Rate for run:  '+str(df.pix.shape[0]/duration))
ax[1,1].text(3, 8, 'Count Rate for run:  '+str(df.pix.shape[0]/duration))
ax[1,1].text(2, 6, 'Proton Charge:  '+ str(proton_charge*1e-12)+ "pC")
ax[1,1].text(1, 4, 'Count Rate for run per pC:  '+ str(df.pix.shape[0]/(duration * proton_charge*1e-12)))
ax[1,1].text(1, 2, 'Duration of Run:  '+ str(duration) + '  seconds')
# ax[1,1].text(3, 2, 'Unicode: Institut für Festkörperphysik')
plt.show()


# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
# b=ax1.hist2d(x_b, y_b, bins =[x_bins, y_bins], cmap='Blues')

# ax1.set_xlim([-130.05, -129.95])
# ax1.set_ylim([45.90, 46])
# ax1.set_yticks([45.90, 45.925, 45.95, 45.975, 46])
# ax1.set_xticks([-130.05, -130.025, -130, -129.975, -129.95])
# cb1.set_label('Observations/day', fontsize=10)
# ax1.set_ylabel('Latitude', fontsize=10)
# ax1.set_xlabel('Longitude', fontsize=10)


# import plotly.express as px
# df = px.data.tips()
# fig = px.histogram(df, x="total_bill", histnorm='probability density')
# fig.show()
