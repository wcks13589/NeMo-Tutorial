DEFAULT_YAML_CONVERSION_PROMPT_TEMPLATE = "The following document contains a list of items. Parse the list of items into a yaml list of strings. Do not parse any other part of the document. There should be no additional formatting to your response, just the yaml list of strings.\n\n {llm_response}"

DEFAULT_MACRO_TOPICS_PROMPT_TEMPLATE = """\
您可以生成{n_macro_topics}個涵蓋日常生活、世界和科學各個方面的綜合主題嗎？你的答案應該是一個主題列表。盡量讓主題多樣化。輸出格式為：
1. 食物和飲料。
2. 科技。\
"""

DEFAULT_SUBTOPICS_PROMPT_TEMPLATE = "您可以生成{n_subtopics}個涵蓋{macro_topic}各個方面的全面主題嗎？您的答案應該是一個主題列表。盡量讓主題多樣化。"

DEFAULT_OPEN_QA_FROM_TOPICS_PROMPT_TEMPLATE = "你可以生成{n_openlines}個與{topic}相關的問題或要求嗎？這些問題和要求應該儘可能多樣化，並以繁體中文書寫。你的答案應該是一個列表。"

DEFAULT_REVISE_OPEN_QA_PROMPT_TEMPLATE = """\
問題：{openline}\n你可以修改上面的問題，加入更多的背景或細節嗎？修改後的問題可以是以下任一種：
1. 在原始問題中加入一些背景。這些背景可能會說明問題的重要性，解釋背景知識，或加入其他合理的資訊。
2. 將問題改成不同的格式或風格，例如：命令句、答案的長度要求等。
3. 將問題延伸，需要對特定的主題進行闡述或討論某一點。
4. 其他與原始問題相關的問題或陳述。
修改後的問題應該包含兩到四個句子。你應該生成{n_revisions}個修改後的問題或陳述，並將它們列在清單中。請盡量讓它們多樣化。\
"""

DEFAULT_WRITING_TASK_PROMPT_TEMPLATE = "您可以生成{n_openlines}個任務，每個任務都需要創建一個與{topic}有關的{text_material_type}？每個任務應該簡潔明瞭，僅包含一到兩句話。任務應儘可能多樣化。您的答案應該是一個任務列表。"

DEFAULT_REVISE_WRITING_TASK_PROMPT_TEMPLATE = """\
任務：{openline}
您可以修改上述任務以包含更詳細的要求嗎？這些要求可以是以下任一項：
1. 需要闡述特定主題或討論某一點。
2. 需要包含一些例子、數據點或參考資料。
3. 需要遵循特定的格式或風格，例如不超過300字，包含特定詞彙等。
4. 其他合理的要求以使任務更詳細。
修改後的任務應包含兩、三或四句。您應該生成{n_revisions}個修改後的任務列表。請盡量使任務多樣化。\
"""

DEFAULT_CLOSED_QA_PROMPT_TEMPLATE = """\
文本：{document}

基於上述文本，您能否提出{n_openlines}個問題或任務？這些問題或任務可以是以下任一種：
1.詢問文本中的特定信息；
2.總結、重述或解釋文本；
3.寫出與文本類似的內容；
4.與文本相關的任何其他合理請求。

盡量讓問題或任務多樣化。\
"""

DEFAULT_MATH_MACRO_TOPICS_PROMPT_TEMPLATE = "您可以生成{n_macro_topics}個涵蓋{school_level}數學知識的綜合主題嗎？您的答案應該是一個主題列表。盡量讓主題多樣化。"

DEFAULT_MATH_SUBTOPICS_PROMPT_TEMPLATE = "列出{n_subtopics}個數學主題，涵蓋\"{macro_topic}\"的各個方面。您的答案應該是一個主題列表。盡量讓主題多樣化。"

DEFAULT_MATH_CLASSIFICATION_PROMPT_TEMPLATE = """\
概念\"{entity}\"是否屬於以下類別之一？
- 小學、中學、高中和大學所教導的數學概念。
- 重要的數學公理、定理、演算法、方程式或不等式。
- 代表性的數學問題、函數和應用。

您的答案應該以\"Yes\"或\"No\"開頭。\
"""

MATH_PROBLEM_GENERAL_PROMPT_TEMPLATE = "生成{n_openlines}個與\"{topic}\"相關的數學問題，或可以使用\"{topic}\"來解決的數學問題。您的答案應該是一個問題列表。盡量讓它們多樣化。"

MATH_PROBLEM_BEGINNER_PROMPT_TEMPLATE = "生成{n_openlines}個與\"{topic}\"相關的數學問題，或可以使用\"{topic}\"來解決的數學問題。您的答案應該是一個問題列表。盡量讓它們多樣化。這些問題應該適合剛剛學習\"{topic}\"的初學者。您的答案應該是一個問題列表。盡量讓它們多樣化。"

DEFAULT_PYTHON_MACRO_TOPICS_PROMPT_TEMPLATE = "列出{n_macro_topics}個Python程式語言中的重要概念。"

DEFAULT_PYTHON_SUBTOPICS_PROMPT_TEMPLATE = "列出{n_subtopics}個與\"{macro_topic}\"有關的重要概念，使用Python程式語言。"

DEFAULT_PYTHON_CLASSIFICATION_PROMPT_TEMPLATE = """\
概念\"{entity}\"是否屬於以下類別之一？
- 像迴圈、函數和資料結構等Python程式設計概念。
- Python中的重要函數、物件或程式庫。
- 像線性代數等可以用Python實現的數學概念。
- 像貪婪搜索和動態編程等可以用Python解決的基本演算法或電腦科學問題。

您的答案應該以\"Yes\"或\"No\"開頭。\
"""

PYTHON_PROBLEM_BEGINNER_PROMPT_TEMPLATE = "生成{n_openlines}個與\"{topic}\"相關的{language}程式設計問題。這些問題應該適合剛剛學習\"{topic}\"的初學者。您的答案應該是一個問題列表。盡量讓它們多樣化。"

PYTHON_PROBLEM_INTERMEDIATE_PROMPT_TEMPLATE = "生成 {n_openlines}個與\"{topic}\"相關的{language}程式設計問題。這些問題應該適合於具有\"{topic}\"經驗的中級程式設計人員。您的答案應該是一個問題列表。盡量讓它們多樣化。"

PYTHON_PROBLEM_ADVANCED_PROMPT_TEMPLATE = "生成{n_openlines}個與\"{topic}\"相關的{language}程式設計問題。這些問題應該適合具有\"{topic}\"扎實知識和經驗的高級程式設計人員。您的答案應該是一個問題列表。盡量讓它們多樣化。"

DIALOGUE_NORMAL_USER_TURN_PROMPT_TEMPLATE = """\
以下是使用者與助理之間的對話。
<|對話開始|>
{conversation_history}
<|對話結束|>

基於上述對話，請以使用者的語氣生成一個後續請求或問題。請直接給出問題，不要額外的文字。\
"""

DIALOGUE_COMPLEX_USER_TURN_PROMPT_TEMPLATE = """\
以下是使用者與助理之間的對話。
<|對話開始|>
{conversation_history}
<|對話結束|>

基於上述對話，請以使用者的語氣生成一個後續請求或問題。請確保問題足夠複雜和多樣，並適合用作後續問題。請直接給出問題，不要附加多餘的文字。\
"""

DIALOGUE_CONCISE_USER_TURN_PROMPT_TEMPLATE = """\
以下是使用者與助理之間的對話。
<|對話開始|>
{conversation_history}
<|對話結束|>

基於上述對話，請以使用者的語氣生成一個後續請求或問題。請批判性地提出問題。確保問題簡潔明瞭，具有現實感。請直接給出問題，不要多餘的詞句。\
"""

# Nemotron-CC prompts

NEMOTRON_CC_SYSTEM_PROMPT = "一個具有好奇心的使用者與人工智慧助理之間的對話。助理對於問題給出了有幫助、詳盡和禮貌的答案。"

NEMOTRON_CC_DISTILL_SYSTEM_PROMPT = "你是一個人工智慧助理。你小心地提供準確、事實、深思熟慮、細膩的答案，並且在推理方面非常出色。"

WIKIPEDIA_REPHRASING_PROMPT_TEMPLATE = """\
以下段落請提供多樣化的同義轉述版本，使用高品質的繁體中文，與維基百科的句子風格相同。請在答案的第一行以\"Here is a paraphrased version:\"作為開頭。

文本：
{document}\
"""

DIVERSE_QA_PROMPT_TEMPLATE = """\
任務：
閱讀文本，提出問題並回答。

按照以下指示進行：
1. 提出多樣化的問題，需要不同認知技能或涵蓋文本的不同方面。
2. 以多種形式提問，例如：
  - 需要確定陳述是否為真或假的是/否問題。
  - 問題以「什麼」、「如何」、「何時」、「何地」、「為什麼」和「誰」等詞語開頭的開放式問題。
  - 提供兩個或更多選項供選擇的多選問題。請在問題中包含選項。
  - 比較兩個數量或物體並確定它們之間關係的比較問題。
  - 測試理解和分析文本能力的閱讀理解問題。
  - 測試解決數學、物理或邏輯問題能力的問題解決問題。
3. 集中提問文本中的事實資訊、重要知識或具體細節。
4. 使用清晰簡潔的語言書寫問題和答案。
5. 使用純文本。請勿使用Markdown格式。
6. 每個問題和答案對應應在單獨一行上。註記問題為「問題：」，答案為「答案：」。

文本：
{document}

任務：
閱讀上述文本後，提出最多8個問題並提供正確答案，按照指示進行。請以以下格式提交您的回應：

以下是基於提供文本的問題和答案：
- 問題：[第一個問題] 答案：[第一個答案]
- 問題：[第二個問題] 答案：[第二個答案]
....\
"""

DISTILL_PROMPT_TEMPLATE = """\
您的任務是根據以下指示閱讀和改寫所提供的文本：
- 旨在創建一個簡潔但準確和豐富的原始文本版本，而不是一個簡單的摘要。
- 捕捉和儲存原始文本中的關鍵資訊、關鍵概念、重要價值、事實細節，同時使其更易於閱讀和訪問。
- 保留專業術語、專業詞彙和複雜概念。
- 保留例子、推理過程的解釋和支持證據，以維持文本的深度和背景。
- 僅包含原始文本中存在的資訊。不要添加新的或無法證實的主張。
- 以純文本格式寫入文本，而不進行格式化。

以下是文本：
{document}

任務：
仔細閱讀上述文本後，按照指示用高品質和清晰的繁體中文改寫它。請以\"Paraphrased Text:\"開頭。\
"""

EXTRACT_KNOWLEDGE_PROMPT_TEMPLATE = """\
您的任務是根據以下指示，從提供的文本中重寫知識。
- 使用易於理解且品質高的繁體中文重寫文本，例如教科書和維基百科中的句子。
- 將重點放在人文、社會科學、自然科學、技術、工程、數學、法律、商業、管理、藝術、教育、農業科學、政治和歷史等學科的內容上。
- 忽略不包含有用事實或知識的內容。
- 保留例子、推理過程的解釋和支持證據，以維持文本的深度和背景。
- 不要添加或修改細節。只需重述文本中已有的內容。
- 使用純文本書寫。
- 不要添加標題、副標題、註釋或評論。

文本：
{document}

任務：
根據指示，從上述文本中重寫事實和知識，形成一篇或多篇文章。\
"""

KNOWLEDGE_LIST_PROMPT_TEMPLATE = """\
審查文本並提取關鍵資訊。按照以下指示進行：
- 仔細閱讀上述文本，並提供一份簡潔且結構良好的列表，列出從文本中提取的實際資訊、具體細節、關鍵概念和重要數字和統計數據。
- 確保每一點都清晰、具體且得到原始文本的支持。
- 確保提取的文本資訊密度高，易於學習。
- 不要添加額外的標題。

文本：
{document}

任務：
按照指示從上述文本中提取事實資訊、具體細節和關鍵概念。\
"""
