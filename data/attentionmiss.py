import os
import io
import json
import unicodedata

x = {"title": "Victoria_(Australia)", "paragraphs": [{"context": "In 1854 at Ballarat there was an armed rebellion against the government of Victoria by miners protesting against mining taxes (the \"Eureka Stockade\"). This was crushed by British troops, but the discontents prompted colonial authorities to reform the administration (particularly reducing the hated mining licence fees) and extend the franchise. Within a short time, the Imperial Parliament granted Victoria responsible government with the passage of the Colony of Victoria Act 1855. Some of the leaders of the Eureka rebellion went on to become members of the Victorian Parliament.", "qas": [{"answers": [{"answer_start": 171, "text": "British troops"}, {"answer_start": 171, "text": "British troops"}, {"answer_start": 171, "text": "British troops"}], "question": "What armed group stopped the uprising at Ballarat?", "id": "570d4c3bfed7b91900d45e33"}]}]}
y = {'version': '1.1', 'data': x}

print x


#with io.open('attention-miss.json', 'w', encoding='utf-8') as f:
#	f.write(unicode(json.dumps(y, ensure_ascii=False)))
