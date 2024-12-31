from openai import OpenAI
import time

class Model:
    def __init__(self, openai_api_key, openai_api_base, model_name):
        self.client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )
        self.model_name = model_name

    def generate(self, messages, temperature=0.0, stop=None):
        # time_start = time.time()
        chat_response = self.client.chat.completions.create(
            # model="llama-3-1-8b-instruct",
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            n=1,
            max_tokens=4096,
            stop=stop
        )

        # print("Time taken:", time.time() - time_start)
        # print("\n\nChat response text:", chat_response.choices[0].message.content)

        return chat_response.choices[0].message.content

class Planner:
    def __init__(self, user_message, model):
        self.user_message = user_message
        self.system_prompt = f'''You are a planner for an assistant trying to solve a user query. You will be given the user query and the assistant's chain of thought so far.

Your task is to 1) rate the assistant's latest thought so far and rate it as '+' (great), '0' (okay), or '-' (bad), and 2) help refine the assistant's next thoughts by generating the next task for the assistant to think about to maximize the probability that the assistant's final answer is correct and minimize risks of the assistant's final answer being incorrect.

A Great '+' thought is anything a good student of math would try. Most of the time it's a clear cut step forward towards solving the problem. But it could also be a sub-optimal choice, as long as it looks like something a reasonably smart human might say while trying to solve the problem.

An Okay '0' thought is anything that's reasonable for a person to say and may be leading to the correct solution, but it's not clear if it's fully correct yet and you'd need to wait for further thoughts to tell if it's great or bad thought.

A Bad '-' thought is one that confidently says something incorrect, is off-topic/weird, leads the solution into a clear dead-end, or is not explained clearly enough for a human to follow along with (even if it is correct).

Please respond in the following format:
Assistant's latest thought: <restate what the assistant's latest thought is>
Observations: <provide observations about the assistant's chain-of-thought so far>
Reasoning: <based on the observations, reason what would be an appropriate rating for the assistant's latest thought and why>
Rating: <based on the reasoning result, provide a rating of '+', '0', or '-', in the format of '+', '0', or '-' in a single line to reflect the latest thought rating.>
Next task: <provide the next task for the assistant to think about>'''
        
        self.model = model

    def continue_planning(self, assistant_thoughts):
        user_prompt = f"USER QUERY: {self.user_message}\n\nASSISTANT CURRENT CHAIN-OF-THOUGHT:\n"

        thoughts = "''"
        if assistant_thoughts:
            thoughts = ""
            for i, thought in enumerate(assistant_thoughts):
                thoughts += f"Thought {i+1}. {thought}\n\n"
            thoughts.strip()
            user_prompt += thoughts
        # print(f"- Planner user prompt -\n\n{user_prompt}\n\n")

        messages=[
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]

        planner_response = self.model.generate(messages)

        # print(f"\nPlanner response: {planner_response}")

        return planner_response


class Assistant:
    def __init__(self, user_message, model, few_shots=0):
        self.user_message = user_message
        self.few_shots = few_shots
        self.few_shot_examples = []

        example_names = ["cipher", "coding", "math", "crossword", "english", "science", "safety", "health_science"]

        for i in range(self.few_shots):
            name = example_names[i]

            with open(f"few_shot_examples/{name}/question.txt", "r", encoding="utf-8") as f:
                question = f.read()

            with open(f"few_shot_examples/{name}/response.txt", "r", encoding="utf-8") as f:
                response = f.read()

            self.few_shot_examples.extend([{
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                ],
            }
            ,
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": response},
                ],
            }
            ]) # TODO: try to split on new line to mimic thought turns as well!

            # response_thoughts = response.split("\n\n")
            # for i, thought in enumerate(response_thoughts):
            #     self.few_shot_examples.append({
            #         "role": "assistant",
            #         "content": [
            #             {"type": "text", "text": thought},
            #         ],
            #     })

        self.system_prompt =f'''Please reason step by step, and put your final answer within \\boxed{{}}.'''
        self.model = model

    def continue_thinking(self, conversation, temperature=0.0, stop="\n\n"):
        messages = [{ "role": "system", "content": [{"type": "text", "text": self.system_prompt}] }] + self.few_shot_examples + conversation
       
        assistant_response = self.model.generate(messages, temperature, stop=stop)

        # print(f"\nAssistant response: {assistant_response}")

        return assistant_response
