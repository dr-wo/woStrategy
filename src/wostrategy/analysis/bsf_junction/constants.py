"""R&D-4 closure contract. Provenance classes are not confidence grades."""
TEAMS = ['Alpine', 'Aston Martin', 'Audi', 'Cadillac', 'Ferrari', 'Haas F1 Team',
         'McLaren', 'Mercedes', 'Racing Bulls', 'Red Bull Racing', 'Williams']
METHOD = 'projected_mean'
VERSION = 'rd4h_projected_mean_v1'
TITLE = 'BSF Junction Engineering Reference — Event Consensus with Explicit Fallback'
METHODOLOGY = (
    'Frozen projected_mean combines matched local observations through explicit full '
    'donor-mean junction transfers, projects reachable supports through the canonical '
    'reference functional, uses canonical or neutral-prior fallback where required, '
    'and centers the completed eleven-team field to mean zero in seconds per lap.'
)
LIMITATION = (
    'Unvalidated engineering reference. Every score depends on canonical projection, '
    'fallback or a prior; support and reference sensitivity vary by event. This is '
    'not equal-fuel pace, age-zero pace, pure car performance, driver-independent '
    'capability or a validated physical decomposition. No-clean-data values are '
    'numerical completion only; fallback-heavy values and changes across support '
    'classes must not be read as measured performance differences. There are no '
    'calibrated team-score confidence intervals.'
)
CAPTION = (
    'Experimental companion reference using local/junction consensus with explicit '
    'fallback. Negative values are faster than Mercedes. P/F/N denotes projected '
    'support, canonical fallback and prior-only completion respectively. This is '
    'not equal-fuel, age-zero or pure car performance.'
)
