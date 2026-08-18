"""Generate the probe-frontier figure for IV-E: precision vs rollback, top-5 acceptance,
comparing the entropy gate, the amendable gate, and the learned probe. Matches the paper's
Fig. 3 style ('better = up and to the left').

Numbers are the held-out top-5 results already computed:
  entropy:   thr .30 -> 54.8/3.6 ; .50 -> 15.4/53.1 ; .70 -> 7.9/93.4
  amendable: m4 -> 72.8/0.9 ; m3 -> 62.6/2.4 ; m2 -> 44.7/8.5
  probe:     .90 -> 75.4/1.5 ; .80 -> 62.4/4.6 ; .30 -> 29.6/32.1
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (rollback%, precision%) points per method, ordered
entropy   = [(3.6,54.8),(53.1,15.4),(93.4,7.9)]
amendable = [(0.9,72.8),(2.4,62.6),(8.5,44.7)]
probe     = [(1.5,75.4),(4.6,62.4),(32.1,29.6)]
labels_e  = ["η=.30","η=.50","η=.70"]
labels_a  = ["m=4","m=3","m=2"]
labels_p  = [".90",".80",".30"]

fig, ax = plt.subplots(figsize=(6.2,5.0))
def plot(points, color, marker, name, labels, off=(6,4)):
    xs=[p[0] for p in points]; ys=[p[1] for p in points]
    ax.plot(xs,ys,marker=marker,color=color,label=name,lw=1.8,ms=7,alpha=0.9)
    for (x,y),t in zip(points,labels):
        ax.annotate(t,(x,y),textcoords="offset points",xytext=off,
                    fontsize=8,color=color)

plot(entropy,  "#C0392B","s","Entropy (confidence)",          labels_e, off=(6,-12))
plot(amendable,"#2E7D32","o","Amendable (convergence)",       labels_a, off=(-30,4))
plot(probe,    "#6A1B9A","D","Learned probe (calibrated)",    labels_p, off=(8,-2))

ax.set_xlabel("Rollback rate (%)")
ax.set_ylabel("Commit precision (%)")
ax.set_title("Commit criteria under top-5 acceptance (held-out)\nbetter = up and to the left")
ax.grid(alpha=0.3)
ax.legend(loc="upper right", framealpha=0.9)
ax.set_xlim(-3, 100); ax.set_ylim(0, 85)

# annotate the key finding: probe & amendable trace the same frontier
ax.annotate("probe and amendable\ntrace the same frontier",
            xy=(3.0,68), xytext=(22,74),
            fontsize=8.5, color="#333",
            arrowprops=dict(arrowstyle="->",color="#777",lw=1))

out="/mnt/user-data/outputs/fig_probe_frontier.png"
plt.savefig(out,dpi=170,bbox_inches="tight")
plt.savefig(out.replace(".png",".pdf"),bbox_inches="tight")   # vector for the dissertation
print("[out]",out)
print("[out]",out.replace(".png",".pdf"))
