import bot_responses

run_bot = True
exit_inputs = set(['exit', 'quit', 'q'])

## TODO: add what can I ask you intent
## TODO: add exit condition
print ("Hello!")
while (run_bot):
    input = raw_input("\n>> ")
    if (input.lower() in exit_inputs):
        run_bot = False
        print ('Goodbye!')
        break
    print(bot_responses.response(input))
