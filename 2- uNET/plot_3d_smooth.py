"""
Referans görsele göre güncellenmiş 3D plot fonksiyonu.
- Pürüzsüz yüzeyler (Gaussian smoothing)
- Organ adı + koordinat tooltip (Parotid_R, x: 189, y: 228, z: 162)
- Gridli eksenler
- Smooth shading

pddca_model_prediction_3d.ipynb içindeki plot_organ_surfaces_3d fonksiyonunu
aşağıdaki ile değiştirin.
"""

# extract_surface fonksiyonunu güncelleyin — Gaussian smoothing ekleyin:
def extract_surface(binary_vol, step_size=1, smooth_sigma=1.0):
    """Pürüzsüz yüzey için maskeye Gaussian blur uygula."""
    import numpy as np
    from skimage.measure import marching_cubes
    smoothed = ndi.gaussian_filter(binary_vol.astype(np.float32), sigma=smooth_sigma)
    smoothed = (smoothed > 0.3).astype(np.float32)  # threshold
    try:
        verts, faces, _, _ = marching_cubes(smoothed, 0.5, step_size=step_size)
        return verts, faces
    except Exception:
        return None, None


# plot_organ_surfaces_3d fonksiyonunu güncelleyin:
def plot_organ_surfaces_3d(label_vol, title='3D Organ Yüzeyleri', organs_to_show=None, step_size=1):
    if organs_to_show is None:
        organs_to_show = list(range(1, 10))
    traces = []
    for c in organs_to_show:
        mask = (label_vol == c).astype(np.uint8)
        if mask.sum() < 10:
            continue
        verts, faces = extract_surface(mask, step_size=step_size, smooth_sigma=1.0)
        if verts is None:
            continue
        col = COLORS_3D[c-1]
        organ_name = STRUCTURE_NAMES[c-1]
        traces.append(go.Mesh3d(
            x=verts[:, 0].tolist(), y=verts[:, 1].tolist(), z=verts[:, 2].tolist(),
            i=faces[:, 0].tolist(), j=faces[:, 1].tolist(), k=faces[:, 2].tolist(),
            color=_rgb(col[:3]), opacity=0.85, name=organ_name,
            flatshading=False,  # Pürüzsüz gradient
            hoverinfo='name',
            hovertemplate=f'<b>{organ_name}</b><br>x: %{{x:.0f}}<br>y: %{{y:.0f}}<br>z: %{{z:.0f}}<extra></extra>',
        ))
    layout = go.Layout(
        title=title,
        height=1000,
        scene=dict(
            xaxis=dict(title='X', showgrid=True, gridcolor='lightgray', backgroundcolor='white'),
            yaxis=dict(title='Y', showgrid=True, gridcolor='lightgray', backgroundcolor='white'),
            zaxis=dict(title='Z', showgrid=True, gridcolor='lightgray', backgroundcolor='white'),
            bgcolor='white',
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )
    fig = go.Figure(data=traces, layout=layout)
    fig.show(config=dict(scrollZoom=True))
