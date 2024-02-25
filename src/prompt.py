
# ref/modify from https://github.com/samrawal/llama2_chat_templater/blob/main/prompt_template.py

class LLamaPromptTemplate:
    def __init__(self, system_prompt=None):
        self.system_prompt = system_prompt
        self.user_messages = []
        self.model_replies = []

    def add_user_message(self, message: str, return_prompt=True):
        self.user_messages.append(message)
        if return_prompt:
            return self.build_prompt()

    def add_model_reply(self, reply: str, includes_history=True, return_reply=True):
        reply_ = reply.replace(self.build_prompt(), "") if includes_history else reply
        self.model_replies.append(reply_)
        if len(self.user_messages) != len(self.model_replies):
            raise ValueError(
                "Number of user messages does not equal number of system replies."
            )
        if return_reply:
            return reply_

    def get_user_messages(self, strip=True):
        return [x.strip() for x in self.user_messages] if strip else self.user_messages

    def get_model_replies(self, strip=True):
        return [x.strip() for x in self.model_replies] if strip else self.model_replies

    def build_prompt(self):

        if self.system_prompt is not None:
            SYS = f"[INST] <<SYS>>\n{self.system_prompt}\n<</SYS>>"
        else:
            SYS = ""

        CONVO = ""
        SYS = "<s>" + SYS
        
        if len(self.user_messages) == len(self.model_replies) + 1:
            len_qa=len(self.user_messages) - 1
        elif len(self.user_messages) == len(self.model_replies):
            len_qa=len(self.user_messages)
        else:
             raise ValueError(
                "Error: Expected {len(user_messages) = len(model_replies) + 1} or {len(user_messages) = len(model_replies)}."
            )
        for i in range(len_qa):
            user_message, model_reply = self.user_messages[i], self.model_replies[i]
            conversation_ = f"{user_message} [/INST] {model_reply} </s>"
            if i != 0 or self.system_prompt is None:
                conversation_ = "[INST] " + conversation_
            CONVO += conversation_

        if len(self.user_messages)==1 and len(self.model_replies)==0:
            if self.system_prompt is not None:
                CONVO += f" {self.user_messages[-1]} [/INST]"
            else:
                CONVO += f"[INST] {self.user_messages[-1]} [/INST]"
        elif len(self.user_messages) == len(self.model_replies) + 1:
            CONVO += f"[INST] {self.user_messages[-1]} [/INST]"
            

        return SYS + CONVO