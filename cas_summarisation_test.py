
## Below snippet is only necessary if running on virtual machine to get around proxy ##
import requests
import pypac

def request(method, url, **kwargs):
    with pypac.PACSession() as session:
        return session.request(method=method, url=url, **kwargs)

requests.request = request


## Tested working example taken from https://towardsdatascience.com/how-to-perform-abstractive-summarization-with-pegasus-3dd74e48bafb ## 

from transformers import PegasusForConditionalGeneration as pegcondgen
from transformers import PegasusTokenizer as pegtoken

import torch

# Some complaint text
text = """sorry for not replying sooner but it has taken a week to catch up on my sleep and disconnect from the stress. thank you for your very comprehensive email, the contents of which are noted with thanks.i cannot express enough my gratitude to you for taking the time to actually meet with me and listen fully to my concerns. after having been fighting for 6 months for someone to just hear what i was saying and offer an independent investigation was an enormous relief. that you understood my serious concerns for the safeguarding of students at hazel grove high school almost immediately improved my mental health, i had previously felt impotent, despondent and as though i was going mad. i can now feel as though i have done my very best and can leave it in the hands of the appropriate agencies and regulatory bodies to ensure any and all necessary actions are taken. i haven't received any contact so far from any of the agencies or regulatory bodies you mention and it does concern me that mark sibson was still involved in strategy meetings with the police, lado and local authority as late as last week where questions were formulated by those present to be put to me by inspector gina brennand 2 days later at a separate meeting. it would appear to me that sibson is still at the heart of the decision making within these strategy meetings and party to my whistleblowing disclosures. the fact that i have not been invited to attend any meetings either as a whistleblower or the designated safeguarding lead at hazel grove high school can only lead me to assume that he is still managing to be protected by convincing those around him that i am 'the problem'. an assumption further strengthened by his decision to notify me that i am to be made redundant with the incredulous reason given that safeguarding needs to be strengthened and made more 'robust' at hghs in light of lowe's recent arrest and remand in custody. i am very concerned that this latest move could be viewed, by those not party to my disclosures, as an indication that i am somehow responsible for lowe's conduct and the lax safeguarding at the school. another alarming incident was brought to my attention on friday when a colleague informed me that the husband of one of hghs's other assistant head teachers, mrs mel majid (who has now changed her name to miss sham) was found guilty of offences against children. mr majid was a teacher at stockport school but was suspended in september 2016 following allegations by pupils. i realise this conviction is of mrs majid's husband but for me, highlights judgement issues of her and , potentially hazel grove high school and must also raise a question of stockport's safeguarding at la level, yet again. i should also point out that it was mrs majid who was responsible for the line management of medical needs during the toogood incident in july 2016 and she has not received any sanction for her grave failings. i appreciate your offer of ongoing support and advice even though your direct involvement is concluded, your expertise and independent view is greatly valued."""

model_name = 'google/pegasus-xsum'
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = pegtoken.from_pretrained(model_name)
model = pegcondgen.from_pretrained(model_name).to(torch_device)
batch = tokenizer(text, truncation = True, padding = 'longest', return_tensors = "pt").to(torch_device)
translated = model.generate(**batch)
tgt_text = tokenizer.batch_decode(translated, skip_special_tokens = True)

# display text summary
tgt_text