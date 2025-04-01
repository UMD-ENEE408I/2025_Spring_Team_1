# Python program to translate
# speech to text and text to speech

mode = 0
import speech_recognition as sr
import pyttsx3 

# Initialize the recognizer 
r = sr.Recognizer() 

# Function to convert text to
# speech
def SpeakText(command):
    
    # Initialize the engine
    engine = pyttsx3.init()
    engine.say(command) 
    engine.runAndWait()
    
    
# Loop infinitely for user to
# speak
with sr.Microphone() as source2:
    r.adjust_for_ambient_noise(source2, duration=1)
    while(1):    
    
    # Exception handling to handle
    # exceptions at the runtime
        try:
        
        # use the microphone as source for input.
        
            
            # wait for a second to let the recognizer
            # adjust the energy threshold based on
            # the surrounding noise level 
            
            #listens for the user's input 
            audio2 = r.listen(source2)
            
            # Using google to recognize audio
            MyText = r.recognize_google(audio2)
            MyText = MyText.lower()

            if MyText == "stay":
                if mode == 1:
                    mode = 0
                    print("Wander mode")
                else: 
                    mode = 1
                    print("Stay mode")
            elif MyText == "follow":
                if mode == 2:
                    mode = 0
                    print("Wander mode")
                else: 
                    mode = 2
                    print("Follow mode")
            elif MyText == "retrieve":
                if mode == 3:
                    mode = 0
                    print("Wander mode")
                else: 
                    mode = 3
                    print("Retrieve mode")
            else:
                print("Did you say " + MyText)
            SpeakText(MyText)
            
        except sr.RequestError as e:
            print("Could not request results; {0}".format(e))
        
        except sr.UnknownValueError:
            print("Say that again?")