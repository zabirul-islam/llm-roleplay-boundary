# One-off generator for paper-ncomms/figures/src/fig1_protocol.drawio (editable source; units = pt, 510 pt = 180 mm).
import html, sys
from xml.sax.saxutils import quoteattr

NAVY, BLUE2, RED, GREY, INK = "#0F4D92", "#3775BA", "#B64342", "#7F7F7F", "#222222"
FONT = "fontFamily=Arial;fontColor=%s;" % INK
cells, n = [], [1]
K = 510 / 373.92  # draw.io PDF: 1 unit = 0.733 pt; scale so 510 design-pt -> 180 mm
R = lambda v: round(v * K, 2)


def nid():
    n[0] += 1
    return f"c{n[0]}"


def box(x, y, w, h, label, stroke=GREY, sw=0.5, fill="#FFFFFF", fs=7, align="left", valign="top", dashed=0,
        fontcolor=INK, bold=0, rounded=1):
    i = nid()
    st = (f"rounded={rounded};arcSize=4;absoluteArcSize=1;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};"
          f"strokeWidth={R(sw)};dashed={dashed};dashPattern=3 2;spacingLeft={R(4)};spacingRight={R(4)};spacingTop={R(2)};fontFamily=Arial;fontColor={fontcolor};fontSize={R(fs)};"
          f"align={align};verticalAlign={valign};fontStyle={bold};")
    cells.append(f'<mxCell id="{i}" value={quoteattr(label)} style="{st}" vertex="1" parent="1">'
                 f'<mxGeometry x="{R(x)}" y="{R(y)}" width="{R(w)}" height="{R(h)}" as="geometry"/></mxCell>')
    return i


def text(x, y, w, h, label, fs=7, bold=0, align="left", valign="middle", color=INK):
    i = nid()
    st = (f"text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;fontFamily=Arial;fontSize={R(fs)};fontStyle={bold};"
          f"fontColor={color};align={align};verticalAlign={valign};spacing=0;")
    cells.append(f'<mxCell id="{i}" value={quoteattr(label)} style="{st}" vertex="1" parent="1">'
                 f'<mxGeometry x="{R(x)}" y="{R(y)}" width="{R(w)}" height="{R(h)}" as="geometry"/></mxCell>')
    return i


def edge(src, dst, color=INK, sw=0.75, exit_=None, entry=None):
    i = nid()
    st = (f"edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;endArrow=block;endFill=1;endSize=4;strokeColor={color};"
          f"strokeWidth={R(sw)};")
    if exit_:
        st += f"exitX={exit_[0]};exitY={exit_[1]};exitDx=0;exitDy=0;"
    if entry:
        st += f"entryX={entry[0]};entryY={entry[1]};entryDx=0;entryDy=0;"
    cells.append(f'<mxCell id="{i}" style="{st}" edge="1" parent="1" source="{src}" target="{dst}">'
                 f'<mxGeometry relative="1" as="geometry"/></mxCell>')


def title(x, y, letter, t, w=300):
    text(x, y, 10, 11, letter, fs=9, bold=1)
    text(x + 11, y, w, 11, t, fs=7.5, bold=1)


# ---------------- a: four-block split
title(0, 0, "a", "Four-block split, before any model run", w=190)
CX1, CX2, CW, RY1, RY2, RH = 50, 126, 72, 34, 100, 60
text(CX1, 18, CW, 12, "Calibration items", fs=7, align="center", color=GREY)
text(CX2, 18, CW, 12, "Held-out items", fs=7, align="center", color=GREY)
text(0, RY1, 46, RH, "Calibration students", fs=7, align="right", color=GREY)
text(0, RY2, 46, RH, "Held-out students", fs=7, align="right", color=GREY)
box(CX1, RY1, CW, RH, "<b>cc</b><br>2PL fit, rating map, text baselines")
box(CX2, RY1, CW, RH, "<b>ch</b><br>Sealed; never read", fill="#F2F2F2", dashed=1, fontcolor="#555555")
box(CX1, RY2, CW, RH, "<b>hc</b><br>Anchor answers → each student's ability")
box(CX2, RY2, CW, RH, "<b>hh</b><br>Evaluation; opened after hash lock and OSF record; reads logged",
    stroke=NAVY, sw=1.25)

# ---------------- b: arms -> metric -> H1
BX = 206
title(BX, 0, "b", "Same held-out student–item cells, same model", w=300)
LW = 150
rp = box(BX, 18, LW, 78, "<b>Role-play</b>: Qwen3.5-27B answers each item as a specific held-out student")
a2 = box(BX + 6, 47, LW - 12, 20, "<b>A2</b> (H1 arm): stated accuracy from the student's hc ability", stroke=NAVY,
         sw=1.25, valign="middle", fs=6.5)
box(BX + 6, 71, LW - 12, 18, "<b>A1</b> population ability · <b>A0</b> fixed level", stroke=BLUE2, valign="middle",
    fs=6.5)
dr = box(BX, 104, LW, 50, "<b>Direct rating</b> (same model)")
b4 = box(BX + 6, 124, LW - 12, 22, "<b>B4</b>: “% of students correct” rating, isotonic map to proportion",
         stroke=RED, sw=1.25, valign="middle", fs=6.5)
box(BX, 162, LW, 18, "<b>B1–B3</b> text baselines (2PL, embedding, TF-IDF)", valign="middle", fs=6.5)
MX, MW = BX + LW + 12, 60
met = box(MX, 70, MW, 56, "Spearman ρ with real proportion correct on hh", align="center", valign="middle")
HX = MX + MW + 12
h1 = box(HX, 62, 510 - HX, 72, "<b>H1</b><br>Δρ = ρ(A2) − ρ(B4)<br>H0: Δρ ≥ 0.10<br>one-sided, α = 0.05",
         stroke=NAVY, sw=1.25, align="center", valign="middle")
edge(a2, met, color=NAVY, sw=1, exit_=(1, 0.5), entry=(0, 0.3))
edge(b4, met, color=RED, sw=1, exit_=(1, 0.5), entry=(0, 0.7))
edge(met, h1, color=INK, sw=1)

# ---------------- c: registration timelines
CY = 196
title(0, CY, "c", "Registration and order of access", w=300)
LX, SX, SW, GAP, RH2 = 0, 104, 91, 14, 26
rows = [("<b>EEDI</b><br>mathematics, 130 items",
         ["Calibration fits and design validation", "<b>Pre-registration v3.1</b> (OSF ygvdf)", "hh opened",
          "<b>H1 confirmatory test</b>"], "confirm"),
        ("<b>DBE-KT22</b><br>databases, 31 items",
         ["Addendum A1 and review", "<b>OSF update</b>", "hh opened", "Descriptive only"], "desc"),
        ("<b>FoundationalASSIST</b><br>mathematics, 249 items",
         ["Addendum A2; validation<br>criterion not met", "<b>OSF update</b>", "hh opened", "Descriptive only"], "desc")]
for r, (lab, steps, kind) in enumerate(rows):
    y = CY + 18 + r * (RH2 + 8)
    text(LX, y, SX - 8, RH2, lab, fs=7)
    prev = None
    for k, s in enumerate(steps):
        x = SX + k * (SW + GAP)
        last = k == len(steps) - 1
        if last and kind == "confirm":
            c = box(x, y, SW, RH2, s, stroke=NAVY, sw=1.25, align="center", valign="middle", fontcolor=NAVY)
        elif last:
            c = box(x, y, SW, RH2, s, dashed=1, align="center", valign="middle", fontcolor="#555555")
        elif k == 1:
            c = box(x, y, SW, RH2, s, stroke=INK, sw=0.9, align="center", valign="middle")
        else:
            c = box(x, y, SW, RH2, s, align="center", valign="middle")
        if prev:
            edge(prev, c, color=GREY, sw=0.75)
        prev = c

xml = ('<mxfile host="drawio"><diagram id="fig1" name="Fig1"><mxGraphModel dx="0" dy="0" grid="0" gridSize="5" '
       'guides="1" page="0" pageScale="1" background="#FFFFFF"><root><mxCell id="0"/><mxCell id="1" parent="0"/>'
       + "".join(cells) + "</root></mxGraphModel></diagram></mxfile>")
open(sys.argv[1], "w").write(xml)
print("cells", len(cells))
