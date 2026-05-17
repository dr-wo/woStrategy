from wostrategy.core.session import Session

year = 2021
round = 21

for session_name in ['FP1', 'FP2', 'FP3']:
    session = Session(year, round, session_name)
    quicklaps = session.quicklaps()
