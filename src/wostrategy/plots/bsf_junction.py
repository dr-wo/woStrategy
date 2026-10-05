"""Accepted R&D-4 closure presentation: P/F points, N only in provenance."""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap, BoundaryNorm
from wostrategy.plots.style_maps import F1_TEAM_COLORS
from wostrategy.analysis.bsf_junction.constants import TEAMS
from wostrategy.analysis.bsf_junction.reference import mercedes_relative_history

NOTE = ('Experimental companion reference. Local/junction consensus with explicit\n'
        'fallback; not equal-fuel, age-zero or pure car performance.')

def season_plot_data(history, field):
    """Expose the evidence-aware plot contract independently of matplotlib."""
    relative = mercedes_relative_history(history)
    output = {}
    for team, rows in relative.groupby('team', sort=True):
        valid = rows[rows.support_flag.ne('N')]
        segments = []
        records = rows.to_dict('records')
        for a, b in zip(records, records[1:]):
            if a['support_flag'] == b['support_flag'] == 'P' and b['round'] == a['round']+1:
                segments.append([(a['round'], a[field]), (b['round'], b[field])])
        output[team] = dict(points=valid[['round', field, 'support_flag']].to_dict('records'),
                            P_segments=segments,
                            prior_rounds=rows.loc[rows.support_flag.eq('N'), 'round'].tolist())
    return output

def plot_bsf_season(history, output_dir, *, season, requested_rounds=None, degraded_rounds=None):
    history = mercedes_relative_history(history)
    ROOT = Path(output_dir)
    ROOT.mkdir(parents=True, exist_ok=True)
    teams = TEAMS
    rounds = sorted(requested_rounds or history['round'].unique().tolist())
    palette = {t: F1_TEAM_COLORS[t] for t in teams}
    data = {(int(z['round']), z['team']): z for z in history.to_dict('records')}
    fields = ['bsf_relative_to_mercedes_s', 'bsf_minus_canonical_relative_to_mercedes_s']
    for r in rounds:
        for t in teams:
            data.setdefault((r,t), {**{f: float('nan') for f in fields}, 'team_support': '—', 'support_flag': '—'})
    for row in data.values():row['team_support'] = row['support_flag']
    flags = np.array([[data[(r,t)]['team_support'] for r in rounds] for t in teams])
    codes = np.array([[{'P':0,'F':1,'N':2,'—':3}[f] for f in line] for line in flags])
    neutral_rounds = [int(r) for r,g in history.groupby('round') if
                      not g.support_flag.eq('P').any() and np.ptp(g.score_s_per_lap) < 1e-12]
    degraded_note = '; '.join(
        f'R{r}: excluded {meta["input_coverage"]["excluded_rows"]}/'
        f'{meta["input_coverage"]["original_rows"]} rows'
        for r, meta in (degraded_rounds or {}).items())
    def mark_degraded(fig):
        if degraded_note:
            fig.text(.09, .985, 'DEGRADED INPUT — ' + degraded_note
                     + ' · reduced evidence coverage; see report warning',
                     ha='left', va='top', fontsize=10, color='#9a3412', fontweight='bold')
    def draw(field, stem, title, ylabel, sign_note):
        # Companion artifacts are file exports; never create a native GUI window
        # or change the canonical plotter's global backend.
        fig = Figure(figsize=(17, 11))
        FigureCanvasAgg(fig)
        mark_degraded(fig)
        ax = fig.add_axes([.09, .435, .69, .42])
        support = fig.add_axes([.09, .14, .69, .20], sharex=ax)
        # R6 is numerical completion, even though Mercedes itself carries F.
        for r in neutral_rounds:
            ax.axvspan(r-.38, r+.38, facecolor='#e9edf2', alpha=.9, zorder=0)
            ax.text(r, .975, f'R{r}: no ranking', transform=ax.get_xaxis_transform(),
                    ha='center', va='top', color='#64748b', fontsize=9)
        zero = ax.axhline(0, color=palette['Mercedes'], linewidth=1.2,
                         linestyle=(0, (5, 3)), alpha=.65, zorder=1)
        plotted = []
        segment_count = 0
        for team in teams:
            values = np.array([float(data[(r, team)][field]) for r in rounds])
            provenance = np.array([data[(r, team)]['team_support'] for r in rounds])
            color = palette[team]
            plotted.extend(values[np.isin(provenance, ['P', 'F'])])
            for k in range(len(rounds)-1):
                if provenance[k] == provenance[k + 1] == 'P' and rounds[k+1] == rounds[k]+1:
                    ax.plot(rounds[k:k + 2], values[k:k + 2],
                            color=color, linestyle=(0, (4, 3)), linewidth=1.25,
                            alpha=.65, zorder=2)
                    segment_count += 1
            for flag, marker in [('P', 'o'), ('F', 's')]:
                mask = provenance == flag
                ax.scatter(np.array(rounds)[mask], values[mask], marker=marker,
                           s=53 if flag == 'P' else 58,
                           facecolors=color if flag == 'P' else 'none',
                           edgecolors='white' if flag == 'P' else color,
                           linewidths=.6 if flag == 'P' else 1.5,
                           alpha=1 if flag == 'P' else .85,
                           zorder=4 if flag == 'P' else 3)
        lo, hi = min(plotted), max(plotted)
        pad = max((hi - lo) * .11, .13)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlim(rounds[0]-.45, rounds[-1]+.45)
        ax.set_xticks(rounds, [f'R{r}' for r in rounds])
        ax.set_ylabel(ylabel, labelpad=12)
        ax.grid(axis='both', color='#e2e8f0', linewidth=.6, alpha=.8)
        ax.set_axisbelow(True)
        fig.text(.09, .944 if degraded_note else .962, title,
                 ha='left', fontsize=21, fontweight='bold')
        fig.text(.09, .907, NOTE, ha='left', fontsize=11, color='#334155', linespacing=1.4)
        fig.text(.09, .865, sign_note, ha='left', fontsize=10, color='#526175')
        handles = [Line2D([0], [0], color=palette[t], marker='o',
                          linewidth=1.2, linestyle=(0, (4, 3)), markersize=5,
                          label=t) for t in teams]
        team_legend = ax.legend(handles=handles, title='Team · canonical colours',
                                loc='upper left', bbox_to_anchor=(1.02, 1.0),
                                frameon=False, fontsize=10, title_fontsize=11,
                                labelspacing=.9, handlelength=2.4, borderaxespad=0)
        ax.add_artist(team_legend)
        evidence = [
            Line2D([0], [0], marker='o', linestyle='', color='#334155', markersize=7,
                   label='P  projected graph support'),
            Line2D([0], [0], marker='s', linestyle='', color='#334155',
                   markerfacecolor='none', markersize=7,
                   label='F  full canonical fallback'),
            Line2D([0], [0], marker='x', linestyle='', color='#64748b', markersize=7,
                   label='N  prior only; strip below'),
        ]
        fig.legend(handles=evidence, loc='upper left', bbox_to_anchor=(.795, .435),
                   frameon=False, fontsize=9.5, handlelength=1.3, labelspacing=1.2)
        fig.text(.801, .322, 'Mercedes is exactly 0\nby definition at every round.',
                 fontsize=10, color='#475569', linespacing=1.6)
        # All 165 provenance flags stay visible; prior-only numeric values have no
        # performance-axis marker or connecting segment.
        support.imshow(codes, cmap=ListedColormap(['#e1edf7', '#fff0d6', '#e5e7eb', '#ffffff']),
                       norm=BoundaryNorm([-.5, .5, 1.5, 2.5, 3.5], 4),
                       extent=(rounds[0]-.5, rounds[-1]+.5, 10.5, -.5), aspect='auto')
        for i, team in enumerate(teams):
            for k, r in enumerate(rounds):
                support.text(r, i, flags[i, k], ha='center', va='center',
                             fontsize=9, color='#475569' if flags[i, k] == 'N' else '#172337',
                             fontweight='bold' if flags[i, k] == 'N' else 'normal')
        support.set_yticks(range(11), teams, fontsize=9)
        for label, team in zip(support.get_yticklabels(), teams):
            label.set_color(palette[team])
        support.set_xticks(rounds, [f'R{r}' for r in rounds], fontsize=10)
        support.set_xticks(np.arange(rounds[0]-.5, rounds[-1]+1, 1), minor=True)
        support.set_yticks(np.arange(-.5, 11, 1), minor=True)
        support.grid(which='minor', color='white', linewidth=1)
        support.tick_params(which='minor', length=0)
        support.tick_params(axis='y', length=0)
        support.set_title('Provenance by team and round · N is excluded from the performance axis',
                          loc='left', fontsize=11, pad=10)
        fig.text(.09, .084,
                 'Dashed links join consecutive P rounds only; F points are isolated. '
                 'N cells represent prior-only numerical completion, not measured pace.',
                 fontsize=10, color='#475569')
        fig.text(.09, .052,
                 'Each contrast also depends on Mercedes’ support (shown in its provenance row). '
                 'Prior-only rounds have no performance ranking; no values are smoothed or clipped.',
                 fontsize=10, color='#475569')
        fig.savefig(ROOT / f'{stem}.png', dpi=180)
        fig.savefig(ROOT / f'{stem}.svg')
        return {'stem': stem, 'performance_markers': int(np.isin(flags, ['P', 'F']).sum()),
                'N_only_in_provenance_strip': int((flags == 'N').sum()),
                'P_to_P_dashed_segments': segment_count,
                'performance_axis_range': [float(lo - pad), float(hi + pad)]}
    tracker_contract = draw('bsf_relative_to_mercedes_s', 'bsf_junction_reference_mercedes',
         f'{season} BSF Junction Engineering Reference Relative to Mercedes',
         'BSF relative to Mercedes (s/lap)',
         'Team − Mercedes · negative = faster · positive = slower · frozen projected_mean')
    diagnostic_contract = draw('bsf_minus_canonical_relative_to_mercedes_s', 'bsf_junction_reference_minus_canonical',
         f'{season} BSF − Canonical: Mercedes-relative Diagnostic',
         'BSF relative − canonical relative (s/lap)',
         '(BSF team − BSF Mercedes) − (canonical team − canonical Mercedes) · positive = BSF places the team slower relative to Mercedes')
    fig = Figure(figsize=(17, 7))
    FigureCanvasAgg(fig)
    mark_degraded(fig)
    ax = fig.subplots()
    fig.subplots_adjust(left=.14, right=.97, top=.84, bottom=.16)
    ax.imshow(codes, cmap=ListedColormap(['#e1edf7','#fff0d6','#e5e7eb','#fff']),
              norm=BoundaryNorm([-.5,.5,1.5,2.5,3.5],4), aspect='auto')
    for i,t in enumerate(teams):
        for k,r in enumerate(rounds):
            row = data[(r,t)]
            ax.text(k,i,f"{flags[i,k]}\n{row.get('direct_local_edge_count', '—')}e · {row.get('projected_support_nodes','—')}n",
                    ha='center',va='center',fontsize=9)
    ax.set_xticks(range(len(rounds)), [f'R{r}' for r in rounds])
    ax.set_yticks(range(11), teams)
    fig.suptitle(f'{season} BSF support / fallback provenance', x=.14,
                 y=.935 if degraded_note else .98, ha='left', fontsize=19)
    fig.text(.14,.88,'P: projected graph support · F: full canonical fallback · N: prior only · —: unavailable event',fontsize=11)
    fig.text(.14,.07,'e = incident local-library edges (overlapping, not independent) · n = projected observed-support nodes',fontsize=10)
    fig.text(.14,.03,'P denotes provenance, not confidence. Every score is fallback-dependent; N is numerical completion only.',fontsize=10)
    fig.savefig(ROOT/'bsf_junction_reference_support.png',dpi=180)
    (ROOT / 'bsf_junction_reference.plot_contract.json').write_text(json.dumps({
        'season': season, 'rounds': rounds, 'teams': teams, 'team_colours': palette,
        'tracker': tracker_contract, 'diagnostic': diagnostic_contract,
        'support_flags': {t: flags[i].tolist() for i,t in enumerate(teams)},
        'unit': 's/lap', 'Mercedes_exact_zero': True,
        'P_is_provenance_not_confidence': True,
        'degraded_rounds': sorted(map(int, degraded_rounds or {})),
    }, indent=2) + '\n')
    return {key:ROOT/f'{stem}.png' for key,stem in {
        'tracker':'bsf_junction_reference_mercedes',
        'diagnostic':'bsf_junction_reference_minus_canonical',
        'support':'bsf_junction_reference_support'}.items()}
