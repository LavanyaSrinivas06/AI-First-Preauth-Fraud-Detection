import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch

# Create figure
fig, ax = plt.subplots(figsize=(11, 6))
ax.axis('off')

# Helper to draw a labeled box
def draw_box(x, y, w, h, label, fontsize=12, linewidth=1.2):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.3",
                         linewidth=linewidth,
                         edgecolor='black',
                         facecolor='white')
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, label, ha='center', va='center', fontsize=fontsize)
    return box

# Positions and sizes
y_mid = 4.5
h = 0.9
w = 1.8
x0 = 0.3
spacing = 2.0

# Draw main components
customer = draw_box(x0, y_mid, w, h, 'Customer\n(Checkout)')
api = draw_box(x0 + spacing, y_mid, w, h, 'Fraud Detection\nAPI')
models = draw_box(x0 + 2*spacing, y_mid, w*1.2, h, 'Models\n(XGBoost + Autoencoder)')
decision = draw_box(x0 + 3*spacing + 0.2, y_mid, w, h, 'Decision Router')
payment = draw_box(x0 + 4*spacing + 0.4, y_mid, w, h, 'Payment\nGateway')

# Analyst and Feedback DB (below Decision and Models)
analyst = draw_box(x0 + 3*spacing + 0.2, y_mid - 1.8, w, h, 'Analyst\nDashboard')
feedback = draw_box(x0 + 2*spacing, y_mid - 1.8, w*1.1, h, 'Feedback Store\n(Database)')

# Draw arrows with labels
def arrow(x1, y1, x2, y2, label=None, text_offset=(0, -0.18)):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle='->', lw=1.6, color='black'))
    if label:
        tx = (x1 + x2) / 2 + text_offset[0]
        ty = (y1 + y2) / 2 + text_offset[1]
        ax.text(tx, ty, label, ha='center', va='center', fontsize=10)

# Coordinates for arrow endpoints (using box centers)
def center(box):
    x, y = box.get_x(), box.get_y()
    w, h = box.get_width(), box.get_height()
    return x + w/2, y + h/2

c_cx, c_cy = center(customer)
a_cx, a_cy = center(api)
m_cx, m_cy = center(models)
d_cx, d_cy = center(decision)
p_cx, p_cy = center(payment)
ana_cx, ana_cy = center(analyst)
fb_cx, fb_cy = center(feedback)

# Main flows
arrow(c_cx + 0.9, c_cy, a_cx - 0.9, a_cy, 'checkout request\n(transaction data)')
arrow(a_cx + 0.9, a_cy, m_cx - 0.9, m_cy, 'features / enriched data')
arrow(m_cx + 0.9, m_cy, d_cx - 0.9, d_cy, 'score / anomaly flag')
arrow(d_cx + 0.9, d_cy, p_cx - 0.9, p_cy, 'approve → forward to PG', text_offset=(0, -0.35))

# Review and block flows
arrow(d_cx, d_cy - 0.45, ana_cx, ana_cy + 0.45, 'uncertain → review', text_offset=(0, 0.06))
arrow(d_cx - 0.6, d_cy - 0.5, d_cx - 1.2, d_cy - 1.4, 'block', text_offset=(-0.2, 0))

# Feedback loop
arrow(ana_cx, ana_cy - 0.45, fb_cx, fb_cy + 0.45, 'analyst label / outcome')
arrow(fb_cx + 0.9, fb_cy + 0.1, m_cx - 0.9, m_cy - 0.1, 'training data / feedback\n(retrain loop)')

# Small legend
legend_x = x0 + 0.1
legend_y = 0.35
ax.text(legend_x, legend_y + 0.25, 'Legend:', fontsize=10, fontweight='bold', ha='left')
ax.text(legend_x, legend_y + 0.05, '- Rectangles = system components', fontsize=9, ha='left')
ax.text(legend_x, legend_y - 0.12, '- All arrows = data / control flow', fontsize=9, ha='left')
ax.text(legend_x, legend_y - 0.29, '- Black-only (print-safe)', fontsize=9, ha='left')

plt.tight_layout()
output_path = 'docs/figures/thesis_diagrams/fig_architecture.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()
print(f'[OK] saved → {output_path}')
