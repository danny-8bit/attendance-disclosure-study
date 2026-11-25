"""Build the report's vector figures and LaTeX tables from the recorded results."""

import csv
import json
from pathlib import Path
from statistics import mean

from reportlab.lib.colors import HexColor, white
from reportlab.pdfgen.canvas import Canvas


ROOT = Path(__file__).resolve().parent
BLUE = HexColor('#205C78')
ORANGE = HexColor('#9A442C')
GREY = HexColor('#52646E')
INK = HexColor('#23333C')
GRID = HexColor('#D8DFE3')
MODES = ('total_only', 'phase_quotas')


def canvas(name, height):
    return Canvas(str(ROOT / 'figures' / f'{name}.pdf'), pagesize=(445, height),
                  pageCompression=1, invariant=1)


def axes(c, bottom, height, xlim, ylim, xticks, yticks, xlabel='', ylabel=''):
    left, width = 56, 372

    def x(value):
        return left + (value - xlim[0]) / (xlim[1] - xlim[0]) * width

    def y(value):
        return bottom + (value - ylim[0]) / (ylim[1] - ylim[0]) * height

    c.setFont('Helvetica', 9)
    for value in yticks:
        yy = y(value)
        c.setStrokeColor(GRID)
        c.setLineWidth(0.45)
        c.line(left, yy, left + width, yy)
        c.setFillColor(INK)
        c.drawRightString(left - 6, yy - 3, f'{value:g}')
    c.setStrokeColor(INK)
    c.setLineWidth(0.65)
    c.line(left, bottom, left, bottom + height)
    c.line(left, bottom, left + width, bottom)
    for value, label in xticks:
        xx = x(value)
        c.line(xx, bottom, xx, bottom - 3)
        c.drawCentredString(xx, bottom - 14, label)
    if xlabel:
        c.drawCentredString(left + width / 2, bottom - 31, xlabel)
    if ylabel:
        c.saveState()
        c.translate(15, bottom + height / 2)
        c.rotate(90)
        c.drawCentredString(0, 0, ylabel)
        c.restoreState()
    return x, y


def series(c, x, y, points, color, dashed=False, markers=False):
    c.setLineWidth(1.7)
    c.setStrokeColor(color)
    c.setFillColor(color)
    if dashed:
        c.setDash(5, 3)
    path = c.beginPath()
    path.moveTo(x(points[0][0]), y(points[0][1]))
    for xx, yy in points[1:]:
        path.lineTo(x(xx), y(yy))
    c.drawPath(path)
    c.setDash()
    if markers:
        for xx, yy in points:
            c.circle(x(xx), y(yy), 2.15, stroke=0, fill=1)


def line_legend(c, baseline, items):
    left = 58
    for label, color, dashed in items:
        c.setStrokeColor(color)
        c.setLineWidth(1.7)
        if dashed:
            c.setDash(5, 3)
        c.line(left, baseline + 3, left + 21, baseline + 3)
        c.setDash()
        c.setFillColor(INK)
        c.setFont('Helvetica', 9)
        c.drawString(left + 26, baseline, label)
        left += 45 + c.stringWidth(label, 'Helvetica', 9)


def attendance_effect(row):
    return 100 * (row['attendance'][1] - row['attendance'][0])


def plot_exact_effect(exact):
    c = canvas('disclosure_effect', 220)
    x, y = axes(c, 48, 139, (0.043, 0.957), (-15, 15),
                [(v, f'{v:g}') for v in (0.05, 0.25, 0.45, 0.625, 0.80, 0.95)],
                range(-15, 16, 5), 'Net attendance cost (penalty units)', 'Disclosure effect (pp)')
    c.setStrokeColor(GREY)
    c.setLineWidth(0.75)
    c.line(x(0.05), y(0), x(0.95), y(0))
    c.setDash(2, 3)
    c.line(x(.625), y(-15), x(.625), y(15))
    c.setDash()
    for rule, color, dashed in [('linear', GREY, True), ('first_warning', BLUE, False)]:
        points = [(row['cost'], attendance_effect(row)) for row in exact if row['rule'] == rule]
        series(c, x, y, points, color, dashed)
    line_legend(c, 202, [('First warning', BLUE, False), ('Linear penalty', GREY, True)])
    c.save()


def plot_timing(main):
    c = canvas('attendance_by_lesson', 326)
    for cost, bottom in [(0.575, 195), (0.65, 46)]:
        c.setFillColor(INK)
        c.setFont('Helvetica-Bold', 10)
        c.drawString(56, bottom + 110, f'Net cost {cost:.3f}')
        x, y = axes(c, bottom, 98, (0.75, 16.25), (0, 102),
                    [(v, str(v)) for v in (1, 4, 8, 9, 12, 16)], (0, 25, 50, 75, 100),
                    'Lesson' if bottom == 46 else '', 'Attendance (%)')
        c.setStrokeColor(GREY)
        c.setDash(2, 3)
        c.line(x(8.5), y(0), x(8.5), y(100))
        c.setDash()
        for mode, color in enumerate((BLUE, ORANGE)):
            points = [(t, 100 * v) for t, v in enumerate(main[cost]['by_lesson'][mode], 1)]
            series(c, x, y, points, color, bool(mode), True)
    line_legend(c, 316, [('Total only', BLUE, False), ('Phase quotas', ORANGE, True)])
    c.save()


def plot_validation():
    with (ROOT / 'results' / 'training_curve.csv').open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    checkpoints = sorted({int(row['semester']) for row in rows})
    c = canvas('validation_learning', 195)
    x, y = axes(c, 48, 112, (-0.1, len(checkpoints) - .9), (-10, -6.5),
                [(idx, f'{v:,}') for idx, v in enumerate(checkpoints)], (-10, -9, -8, -7),
                'Training semesters (checkpoint spacing)', 'Validation utility')
    for mode, color in enumerate((BLUE, ORANGE)):
        points = [(idx, mean(float(r['utility']) for r in rows
                             if int(r['semester']) == checkpoint and r['disclosure'] == MODES[mode]))
                  for idx, checkpoint in enumerate(checkpoints)]
        series(c, x, y, points, color, bool(mode), True)
    line_legend(c, 177, [('Total only', BLUE, False), ('Phase quotas', ORANGE, True)])
    c.save()


def plot_learned_effect(learning, main):
    c = canvas('learned_disclosure_effect', 199)
    x, y = axes(c, 46, 120, (-.35, len(learning) - .65), (-10, 12),
                [(i, f"{row['cost']:.3f}") for i, row in enumerate(learning)], (-10, -5, 0, 5, 10),
                'Net attendance cost (penalty units)', 'Disclosure effect (pp)')
    c.setStrokeColor(GREY)
    c.setLineWidth(.75)
    c.line(x(-.35), y(0), x(len(learning) - .65), y(0))
    for idx, row in enumerate(learning):
        effect = row['attendance_difference_pp']
        a, b = effect['ci95']
        level = y(effect['mean'])
        c.setStrokeColor(ORANGE)
        c.setFillColor(ORANGE)
        c.setLineWidth(1.2)
        xx = x(idx + .09)
        c.line(xx, y(a), xx, y(b))
        c.line(xx - 3, y(a), xx + 3, y(a))
        c.line(xx - 3, y(b), xx + 3, y(b))
        c.circle(xx, level, 2.6, stroke=0, fill=1)
        yy = y(attendance_effect(main[row['cost']]))
        xx = x(idx - .09)
        c.setFillColor(white)
        c.setStrokeColor(BLUE)
        c.rect(xx - 2.5, yy - 2.5, 5, 5, stroke=1, fill=1)
    c.setStrokeColor(BLUE)
    c.rect(58, 181, 5, 5, stroke=1, fill=0)
    c.setFillColor(INK)
    c.setFont('Helvetica', 9)
    c.drawString(70, 180, 'Dynamic programming')
    c.setFillColor(ORANGE)
    c.circle(187, 184, 2.6, stroke=0, fill=1)
    c.setFillColor(INK)
    c.drawString(196, 180, 'Q-learning (paired 95% CI)')
    c.save()


def tex_rows(path, command, rows):
    body = '\n'.join(' & '.join(row) + r' \\%' for row in rows)
    path.write_text(f'\\newcommand{{\\{command}}}{{%\n{body}\n}}\n', encoding='ascii')


def write_latex(parameters, exact, main, linear, learning):
    figures = ROOT / 'figures'
    tex_rows(figures / 'exact_rows.tex', 'StudyExactRows', (
        [f'{cost:.3f}', *(f'{100*v:.1f}' for v in main[cost]['attendance']),
         f"{attendance_effect(main[cost]):+.1f}", f"{attendance_effect(linear[cost]):+.1f}",
         f"{main[cost]['utility'][1] - main[cost]['utility'][0]:.3f}"]
        for cost in [r['cost'] for r in learning]))
    tex_rows(figures / 'learning_rows.tex', 'StudyLearningRows', (
        [f"{r['cost']:.3f}", *(f'{100*v:.1f}' for v in r['attendance']),
         f"{r['attendance_difference_pp']['mean']:+.2f}",
         '[{:+.2f}, {:+.2f}]'.format(*r['attendance_difference_pp']['ci95'])]
        for r in learning))
    tex_rows(figures / 'utility_rows.tex', 'StudyUtilityRows', (
        [f"{r['cost']:.3f}", *(f'{v:.3f}' for v in r['utility']),
         *(f"{v['mean']:.4f}" for v in r['exact_utility_regret'])] for r in learning))
    tex_rows(figures / 'phase_rows.tex', 'StudyPhaseRows', (
        [f'{cost:.3f}', name,
         *(f'{100 * mean(values[start:start + 8]):.1f}'
           for start in (0, 8) for values in source[cost]['by_lesson'])]
        for cost in (0.575, 0.65)
        for name, source in [('Linear', linear), ('First warning', main)]))
    parameters_to_tex = {'ScheduleCount': 'schedule_count', 'StateCount': 'observable_states',
                         'SemesterCount': 'train_semesters', 'TestCount': 'test_size',
                         'ValidationCount': 'validation_size'}
    macros = {name: f'{parameters[key]:,}' for name, key in parameters_to_tex.items()}
    by_cost = {r['cost']: r for r in learning}
    low, high = main[0.575], main[0.65]
    lag = main[0.625]['optimal_attendance_bounds']
    macros.update({
        'SeedCount': str(len(parameters['train_seeds'])),
        'LowExact': f'{attendance_effect(low):+.2f}',
        'HighExact': f'{attendance_effect(high):+.2f}',
        'LowQL': f"{by_cost[0.575]['attendance_difference_pp']['mean']:+.2f}",
        'HighQL': f"{by_cost[0.65]['attendance_difference_pp']['mean']:+.2f}",
        'PivotQL': f"{by_cost[0.625]['attendance_difference_pp']['mean']:+.2f}",
        'PivotExact': f'{attendance_effect(main[0.625]):+.2f}',
        'PivotTotalMin': f'{100 * lag[0][0]:.2f}', 'PivotTotalMax': f'{100 * lag[0][1]:.2f}',
        'PivotQuotaMin': f'{100 * lag[1][0]:.2f}', 'PivotQuotaMax': f'{100 * lag[1][1]:.2f}',
        'MaxMeanRegret': f"{max(g['mean'] for r in learning for g in r['exact_utility_regret']):.4f}",
        'AllPositiveValue': str(sum(r['utility'][1] > r['utility'][0] + 1e-12 for r in main.values())),
        'MinMainDelta': f'{min(attendance_effect(r) for r in main.values()):+.2f}',
        'MaxMainDelta': f'{max(attendance_effect(r) for r in main.values()):+.2f}',
        'NumCostPoints': str(len(exact) // 2),
    })
    (figures / 'report_values.tex').write_text(
        '\n'.join(f'\\newcommand{{\\Data{name}}}{{{value}}}' for name, value in macros.items()) + '\n',
        encoding='ascii')


def main():
    with (ROOT / 'results' / 'summary.json').open(encoding='utf-8') as handle:
        summary = json.load(handle)
    if summary['parameters']['quick']:
        raise ValueError('The report requires results from the full experiment')
    (ROOT / 'figures').mkdir(exist_ok=True)
    exact = summary['exact']
    warning = {row['cost']: row for row in exact if row['rule'] == 'first_warning'}
    linear = {row['cost']: row for row in exact if row['rule'] == 'linear'}
    plot_exact_effect(exact)
    plot_timing(warning)
    plot_validation()
    plot_learned_effect(summary['learning'], warning)
    write_latex(summary['parameters'], exact, warning, linear, summary['learning'])


if __name__ == '__main__':
    main()
