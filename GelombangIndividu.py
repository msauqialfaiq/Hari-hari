import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

fname = input("Nama file Excel: ")
try:
    df = pd.read_excel(fname, sheet_name='Raw Data',
                       usecols='B,F', skiprows=1, header=None, names=['t', 'y'])

    print(df.head())

    # konversi waktu kalo format datetime
    if pd.api.types.is_datetime64_any_dtype(df['t']) or pd.api.types.is_object_dtype(df['t']):
        base_time = pd.to_datetime(df['t'].astype(str)).dt.time
        df['t'] = [t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1e6 for t in base_time]

    df.dropna(inplace=True)
    print(f"Data rows: {len(df)}")

    mean_level = df['y'].mean()
    y_detrended = df['y'] - mean_level

    # cari zero crossing
    up_cross_indices = np.where((y_detrended.iloc[:-1].values < 0) &
                                (y_detrended.iloc[1:].values >= 0))[0]

    wave_heights = []
    wave_periods = []
    wave_crests = []
    wave_troughs = []

    for i in range(len(up_cross_indices) - 1):
        start = up_cross_indices[i]
        end = up_cross_indices[i+1]

        wave_y = df['y'][start : end + 1]
        wave_t = df['t'][start : end + 1]

        if not wave_y.empty:
            crest = wave_y.max()
            trough = wave_y.min()
            H = crest - trough

            wave_heights.append(H)
            wave_crests.append(crest)
            wave_troughs.append(trough)

            # hitung periode
            t1 = np.interp(0, y_detrended[start:start+2], df['t'][start:start+2])
            t2 = np.interp(0, y_detrended[end:end+2], df['t'][end:end+2])
            T = t2 - t1
            wave_periods.append(T)

    # bikin dataframe hasil
    results = pd.DataFrame({
        'Periode (T)': wave_periods,
        'Tinggi (H)': wave_heights,
        'Puncak (Crest)': wave_crests,
        'Lembah (Trough)': wave_troughs
    })

    print("\n=== Hasil Analisis ===")
    print(results)
    print("\n=== Statistik ===")
    print(results.describe())

    # hitung gelombang signifikan
    sorted_h = sorted(wave_heights, reverse=True)
    third = len(sorted_h) // 3
    hs = np.mean(sorted_h[:third])

    closest_idx = np.abs(np.array(wave_heights) - hs).argmin()
    ts = wave_periods[closest_idx]

    print(f"\nHs: {hs:.4f}")
    print(f"Ts: {ts:.4f}")

    rmse = np.sqrt(mean_squared_error(df['y'], np.full_like(df['y'], mean_level)))
    print(f"RMSE: {rmse:.4f}")

    # plot
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df['t'], df['y'], 'b', linewidth=1.2, label='Elevasi')
    ax.axhline(mean_level, color='k', linestyle='--', label='MWL')
    ax.plot(df['t'].iloc[up_cross_indices],
            df['y'].iloc[up_cross_indices],
            'go', markersize=4, label='Zero crossing')

    ax.set_title('Wave Time Series Analysis')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Water Level (m)')
    ax.legend()
    ax.grid(True)
    plt.show()

    # simpan ke excel
    outfile = 'Hasil_Analisis_Gelombang.xlsx'
    with pd.ExcelWriter(outfile) as writer:
        results.to_excel(writer, index=False, sheet_name='Individual Wave')

        summary = pd.DataFrame({
            'Parameter': ['Hs', 'Ts', 'RMSE'],
            'Nilai': [hs, ts, rmse]
        })
        summary.to_excel(writer, index=False, sheet_name='Summary')

    print(f"\nDone. Saved to '{outfile}'")

except FileNotFoundError:
    print("File ga ketemu. Pastikan file ada di folder yang sama atau kasih full path")
except Exception as e:
    print(f"Error: {e}")
