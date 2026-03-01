import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, ConnectionPatch

plt.rcParams.update({'font.size': 13})

fig, ax = plt.subplots(figsize=(15, 6))
ax.axis('off')

# Draw function
def box(x, y, w, h, text, fontsize=13, bold=False):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3", linewidth=2, edgecolor='black', facecolor='white')
    ax.add_patch(p)
    weight = 'bold' if bold else 'normal'
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize, fontweight=weight)
    return p

# Positions
h = 1.1
w = 2.2
x0 = 0.5
gap = 2.5
y_main = 4.5

# Main system components
b_customer = box(x0, y_main, w, h, 'Customer\n(Checkout)', bold=True)
b_api = box(x0 + gap, y_main, w, h, 'Fraud Detection\nAPI', bold=True)
b_models = box(x0 + 2*gap, y_main, w*1.2, h, 'Models\nXGBoost + Autoencoder', bold=True)
b_decision = box(x0 + 3*gap + 0.2, y_main, w, h, 'Decision\nRouter', bold=True)
b_payment = box(x0 + 4*gap + 0.6, y_main, w, h, 'Payment\nGateway', bold=True)
b_block = box(x0 + 3*gap - 1.7, y_main - 1.7, w, h, 'Block\n(Stop Payment)', bold=True)
b_analyst = box(x0 + 3*gap + 0.2, y_main - 1.7, w, h, 'Analyst\nDashboard', bold=True)
b_feedback = box(x0 + 2*gap - 0.2, y_main - 3.2, w*1.1, h, 'Feedback Store\n(Database)', bold=True)

# Helper centers
def center(b):
    x, y = b.get_x(), b.get_y()
    return x + b.get_width()/2, y + b.get_height()/2

c = center(b_customer)
a = center(b_api)
m = center(b_models)
d = center(b_decision)
p = center(b_payment)
blk = center(b_block)
ana = center(b_analyst)
fb = center(b_feedback)

# Arrow drawing helper with label
def arrow(src, dst, label=None, label_offset=(0, 0.18), lw=2.2):
    con = ConnectionPatch(xyA=src, xyB=dst, coordsA='data', coordsB='data', arrowstyle='->', linewidth=lw, color='black')
    ax.add_artist(con)
    if label:
        lx = (src[0] + dst[0])/2 + label_offset[0]
        ly = (src[1] + dst[1])/2 + label_offset[1]
        ax.text(lx, ly, label, ha='center', va='center', fontsize=12)

# Main flows
arrow((c[0]+1.1, c[1]), (a[0]-1.1, a[1]), 'checkout request\n(transaction data)')
arrow((a[0]+1.1, a[1]), (m[0]-1.1, m[1]), 'features / enriched data')
arrow((m[0]+1.1, m[1]), (d[0]-1.1, d[1]), 'score / anomaly flag')
arrow((d[0]+1.1, d[1]), (p[0]-1.1, p[1]), 'approve → Payment Gateway', label_offset=(0, -0.22))

# Block flow
arrow((d[0], d[1]-0.55), (blk[0]+0.7, blk[1]+0.55), 'block / high-risk', label_offset=(0.18, 0))

# Review flow (Decision -> Analyst)
arrow((d[0]+0.2, d[1]-0.55), (ana[0]-0.2, ana[1]+0.55), 'uncertain → review', label_offset=(0.18, 0))
# Analyst to Feedback
arrow((ana[0], ana[1]-0.55), (fb[0]+0.7, fb[1]+0.55), 'analyst label / outcome', label_offset=(0.18, 0))
# Feedback back to Models (retrain loop)
arrow((fb[0]+1.1, fb[1]+0.1), (m[0]-1.1, m[1]-0.2), 'training data / feedback\n(retrain loop)', label_offset=(0, -0.02), lw=1.5)

# Legend
lx = x0 + 0.1
ly = 0.6
ax.text(lx, ly + 0.38, 'Legend:', fontsize=12, fontweight='bold', ha='left')
ax.text(lx, ly + 0.18, '- Boxes = system components', ha='left')
ax.text(lx, ly - 0.02, '- Arrows = data / control flow', ha='left')
ax.text(lx, ly - 0.22, '- Black-only (print-safe)', ha='left')

plt.tight_layout()
PNG_PATH = 'docs/figures/thesis_diagrams/fig_architecture_overview.png'
plt.savefig(PNG_PATH, dpi=300, bbox_inches='tight')
plt.close()
print(f'[OK] saved → {PNG_PATH}')
