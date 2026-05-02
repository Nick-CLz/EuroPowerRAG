import json
from pathlib import Path

articles = [
    {"title": "German wind generation exceeds 50GW, pushing prices negative", "summary": "Strong winds across northern Germany have led to record wind generation, resulting in significant oversupply and negative day-ahead prices for several hours.", "label": -1},
    {"title": "French nuclear fleet reaches 95% availability", "summary": "EDF reports that its nuclear reactors are back online ahead of schedule, providing abundant baseload power and easing supply concerns.", "label": -1},
    {"title": "Mild winter forecast reduces heating demand expectations", "summary": "Meteorologists predict a warmer than average winter for Western Europe, significantly reducing the anticipated demand for gas and electricity heating.", "label": -1},
    {"title": "New solar farms commissioned in Spain add 2GW capacity", "summary": "Several large-scale solar projects have been completed, contributing inexpensive renewable energy to the Iberian grid during daylight hours.", "label": -1},
    {"title": "Gas storage levels reach 99% full across EU", "summary": "European gas storage facilities are near maximum capacity earlier than expected, providing a strong buffer against winter supply shocks and depressing near-term power prices.", "label": -1},
    {"title": "Industrial demand remains sluggish in Q3", "summary": "Manufacturing output in Germany and Italy continues to decline, leading to lower industrial power consumption than historical averages.", "label": -1},
    {"title": "Major interconnector to Norway goes online", "summary": "A new subsea cable connecting the UK grid to cheap Norwegian hydro power has begun commercial operations, increasing supply options.", "label": -1},
    {"title": "Wind output expected to remain strong through weekend", "summary": "Weather models indicate sustained high wind speeds across the North Sea, ensuring robust renewable generation for the coming days.", "label": -1},
    {"title": "Coal plants face accelerated closure timetable", "summary": "However, short-term generation is currently unaffected as ample alternatives are covering demand.", "label": -1},
    {"title": "Carbon allowance prices drop 10% amid oversupply", "summary": "EUA carbon prices fell sharply today, reducing the marginal cost of fossil fuel generation and easing wholesale power prices.", "label": -1},
    
    {"title": "Unplanned outage at French nuclear plant reduces output", "summary": "A reactor has been taken offline unexpectedly due to a cooling system issue, removing 1.2GW of capacity during a high-demand period.", "label": 1},
    {"title": "Cold snap predicted to drive record power demand", "summary": "A sudden drop in temperatures across Europe is expected to cause a spike in electricity usage for heating over the next week.", "label": 1},
    {"title": "Wind speeds drop to seasonal lows", "summary": "A high-pressure system has settled over the North Sea, leading to a dramatic dunkelflaute (dark wind lull) and reducing renewable output.", "label": 1},
    {"title": "Gas prices surge following pipeline disruption", "summary": "A technical issue at a major gas pipeline has curtailed flows into Europe, driving up the cost of gas-fired power generation.", "label": 1},
    {"title": "Carbon price hits new high on stricter EU targets", "summary": "The cost of emitting CO2 has risen significantly, pushing the marginal cost of coal and gas generation higher.", "label": 1},
    {"title": "Delay in new offshore wind farm commissioning", "summary": "A major renewable project expected to come online this month has been delayed due to supply chain issues, keeping supply tighter.", "label": 1},
    {"title": "Heatwave forces nuclear curtailment in France", "summary": "High river temperatures have forced EDF to reduce output at several nuclear plants to comply with environmental regulations.", "label": 1},
    {"title": "Strike action threatens coal deliveries to power stations", "summary": "Rail workers have announced a strike, potentially disrupting fuel supplies to key thermal power plants and raising supply concerns.", "label": 1},
    {"title": "Interconnector failure cuts UK imports from continent", "summary": "A fault on a major subsea cable has halted electricity imports, forcing the grid operator to rely on more expensive domestic generation.", "label": 1},
    {"title": "LNG shipments diverted to Asia amid premium prices", "summary": "Several LNG cargoes originally destined for Europe have been redirected to Asian buyers offering higher prices, tightening the local gas market.", "label": 1},

    {"title": "Energy company reports Q3 earnings", "summary": "E.ON has published its financial results for the third quarter, showing steady profits in line with analyst expectations.", "label": 0},
    {"title": "New CEO appointed at national grid operator", "summary": "The board has announced the appointment of a new chief executive to lead the company's long-term strategy.", "label": 0},
    {"title": "Regulatory body publishes annual market review", "summary": "ACER has released its yearly report summarizing historical trends in the European energy market over the past 12 months.", "label": 0},
    {"title": "Energy minister discusses future transition goals", "summary": "In a speech today, the minister outlined a vision for a carbon-neutral economy by 2050, focusing on long-term policy shifts.", "label": 0},
    {"title": "Utility announces dividend payout to shareholders", "summary": "Following a profitable year, the company board has approved a special dividend payment to investors.", "label": 0},
    {"title": "Conference on smart grid technologies opens in Berlin", "summary": "Industry experts have gathered to discuss the latest innovations in grid management and digitalization.", "label": 0},
    {"title": "Research paper explores hypothetical fusion reactor designs", "summary": "A new academic study models potential efficiency gains in experimental fusion technology that may be viable in the distant future.", "label": 0},
    {"title": "Company launches re-branding campaign", "summary": "The regional utility has unveiled a new logo and marketing slogan focused on sustainability.", "label": 0},
    {"title": "Union negotiations begin over updated worker contracts", "summary": "Management and labor representatives have started scheduled talks regarding minor updates to employee benefits.", "label": 0},
    {"title": "Historic power plant converted into museum", "summary": "A decommissioned coal station from the 1950s has been officially reopened to the public as an educational facility.", "label": 0}
]

out_path = Path("data/eval/sentiment_gold.jsonl")
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    for item in articles:
        f.write(json.dumps(item) + "\n")
print(f"Wrote {len(articles)} items to {out_path}")
