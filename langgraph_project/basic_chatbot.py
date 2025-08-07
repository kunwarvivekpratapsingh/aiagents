import langgraph as lg

# Define a simple state machine for the chatbot
class BasicChatbot(lg.StateMachine):
    def __init__(self):
        super().__init__()
        self.state = 'start'

    def on_message(self, message):
        if self.state == 'start':
            self.state = 'greeting'
            return 'Hello! How can I assist you today?'
        elif self.state == 'greeting':
            if 'search' in message.lower():
                self.state = 'search'
                return self.perform_search(message)
            else:
                self.state = 'end'
                return 'Thank you for your message. Have a great day!'
        else:
            return 'Goodbye!'

    def perform_search(self, query):
        # Simulate a web search
        return fSearch results for {query}: [Simulated search result]

# Instantiate the chatbot
chatbot = BasicChatbot()

# Example interaction
print(chatbot.on_message('Hi'))
print(chatbot.on_message('search for AI news'))
print(chatbot.on_message('I need help'))
