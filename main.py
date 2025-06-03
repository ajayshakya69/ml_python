import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
from groq import Groq
from pydantic import BaseModel, Field, validator
import getpass
import re
from pathlib import Path
import tempfile
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from dotenv import load_dotenv

load_dotenv(".env.local") 



class StudentData(BaseModel):
    """
    Pydantic model for student profile data with validation and normalization.
    Fields are optional and validated for type and range.
    """
    Marks_10th: Optional[float] = Field(None, ge=0, le=100, description="10th standard marks percentage")
    Marks_12th: Optional[float] = Field(None, ge=0, le=100, description="12th standard marks percentage")
    JEE_Score: Optional[int] = Field(None, ge=1, description="JEE score/rank")
    Budget: Optional[int] = Field(None, ge=0, description="Budget for education in rupees")
    Preferred_Location: Optional[str] = Field(None, description="Preferred study location")
    Gender: Optional[str] = Field(None, description="Student's gender")
    Target_Exam: Optional[str] = Field(None, description="Target entrance exams")
    State_Board: Optional[str] = Field(None, description="Educational board")
    Category: Optional[str] = Field(None, description="Reservation category")
    Extra_Curriculars: Optional[str] = Field(None, description="Extracurricular activities")
    Future_Goal: Optional[str] = Field(None, description="Career goals")

    @validator('Budget', pre=True)
    def convert_budget(cls, v):
        """
        Convert budget strings like '5 lakhs' or '2 crore' to integer rupees.
        """
        if v is None:
            return None
        if isinstance(v, str):
            v = v.lower().replace(',', '').replace(' ', '')
            if 'lakh' in v or 'lac' in v:
                num = re.findall(r'\d+\.?\d*', v)
                if num:
                    return int(float(num[0]) * 100000)
            elif 'crore' in v:
                num = re.findall(r'\d+\.?\d*', v)
                if num:
                    return int(float(num[0]) * 10000000)
            else:
                num = re.findall(r'\d+', v)
                if num:
                    return int(num[0])
        return v

    @validator('Gender', pre=True)
    def normalize_gender(cls, v):
        """
        Normalize gender input to 'Male' or 'Female' if possible.
        """
        if v is None:
            return None
        v = str(v).lower()
        if v in ['male', 'boy', 'm', 'man']:
            return 'Male'
        elif v in ['female', 'girl', 'f', 'woman']:
            return 'Female'
        else:
            return v.title()

# =============================
# Main Chatbot Logic Class
# =============================

class CollegeCounselorChatbot:
    """
    Main logic for the AI college counselor chatbot.
    Handles student data collection, LLM extraction, recommendations, and profile management.
    """
    def __init__(self, api_key, name="Lauren"):
        # Initialize chatbot with API key and counselor name
        self.name = name
        self.model = "llama-3.1-8b-instant"
        self.client = Groq(api_key=api_key)
        self.student_data = StudentData()
        self.data_collected = False
        self.recommendations_provided = False
        self.profile_filename = None
        # Create directory for student profiles and initialize a new profile file
        self.profiles_dir = self.create_profiles_directory()
        self.initialize_profile_file()
        # Conversation history for LLM context
        self.conversation_history = [
            {"role": "system", "content": f"""
            You are {name}, an AI college counselor for Indian students. Your goal is to collect information about the student and provide personalized college recommendations.

            You need to gather these key details from the student through natural conversation:
            1. Marks_10th - Student's 10th standard marks percentage
            2. Marks_12th - Student's 12th standard marks percentage
            3. JEE_Score - JEE score if applicable
            4. Budget - How much they can afford for their entire education
            5. Preferred_Location - Which part of India they prefer to study in
            6. Gender - Student's gender
            7. Target_Exam - Which entrance exams they're targeting
            8. State_Board - Which educational board they studied under
            9. Category - Their reservation category (General, OBC, SC, ST, etc.)
            10. Extra_Curriculars - Any extracurricular activities/achievements
            11. Future_Goal - Career aspirations or goals

            Be friendly, conversational, and encouraging. First introduce yourself briefly and start collecting information.
            Only move on to college recommendations after collecting all the necessary information.
            """}
        ]
        # Example college database (can be replaced with a real DB)
        self.colleges = [
            {"name": "IIT Bombay", "min_jee": 8000, "fees": 800000, "location": "Mumbai", "acceptance_rate": "Very Low", "specialties": ["Engineering", "Technology"]},
            {"name": "IIT Delhi", "min_jee": 9000, "fees": 750000, "location": "Delhi", "acceptance_rate": "Very Low", "specialties": ["Engineering", "Computer Science"]},
            {"name": "BITS Pilani", "min_jee": 15000, "fees": 1200000, "location": "Rajasthan", "acceptance_rate": "Low", "specialties": ["Engineering", "Pharmacy"]},
            {"name": "VIT Vellore", "min_jee": 50000, "fees": 900000, "location": "Tamil Nadu", "acceptance_rate": "Moderate", "specialties": ["Engineering", "Bio-Technology"]},
            {"name": "Manipal Institute of Technology", "min_jee": 70000, "fees": 1500000, "location": "Karnataka", "acceptance_rate": "Moderate", "specialties": ["Engineering", "Medicine"]},
            {"name": "NIT Trichy", "min_jee": 20000, "fees": 500000, "location": "Tamil Nadu", "acceptance_rate": "Low", "specialties": ["Engineering"]},
            {"name": "Delhi University", "min_jee": None, "fees": 200000, "location": "Delhi", "acceptance_rate": "Moderate", "specialties": ["Arts", "Commerce", "Science"]},
            {"name": "AIIMS Delhi", "min_jee": None, "fees": 600000, "location": "Delhi", "acceptance_rate": "Very Low", "specialties": ["Medicine"]},
            {"name": "Tula's Institute", "min_jee": 100000, "fees": 600000, "location": "Dehradun", "acceptance_rate": "Moderate", "specialties": ["BCA", "MCA", "BBA", "MBA"]},
            {"name": "Graphic Era University", "min_jee": 100000, "fees": 700000, "location": "Dehradun", "acceptance_rate": "Moderate", "specialties": ["Engineering", "Management", "Computer Science"]},
            {"name": "Doon University", "min_jee": 100000, "fees": 400000, "location": "Dehradun", "acceptance_rate": "Moderate", "specialties": ["Science", "Arts", "Commerce"]},
        ]

    def create_profiles_directory(self):
        """
        Create a directory for storing student profiles. Uses current dir, else temp dir.
        """
        try:
            # Try current directory first
            profiles_dir = Path('./student_profiles')
            profiles_dir.mkdir(parents=True, exist_ok=True)

            # Test write access
            test_file = profiles_dir / 'test_write.txt'
            with open(test_file, 'w') as f:
                f.write('test')
            test_file.unlink()  # Delete test file

            print(f"✅ Profiles directory created/verified at: {profiles_dir.absolute()}")
            return profiles_dir

        except Exception as e:
            print(f"❌ Error with ./student_profiles directory: {e}")
            try:
                # Fallback to temp directory
                import tempfile
                profiles_dir = Path(tempfile.gettempdir()) / 'student_profiles'
                profiles_dir.mkdir(parents=True, exist_ok=True)
                print(f"✅ Using temporary directory: {profiles_dir.absolute()}")
                return profiles_dir
            except Exception as e2:
                print(f"❌ Error creating temp directory: {e2}")
                # Last resort - current directory
                return Path('.')

    def initialize_profile_file(self):
        """
        Create a new profile file for the session with initial empty data.
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.profile_filename = self.profiles_dir / f"student_profile_{timestamp}.txt"

            # Create initial empty profile
            initial_profile = {
                "student_profile": {
                    "session_info": {
                        "profile_created": datetime.now().isoformat(),
                        "counselor_name": self.name,
                        "session_id": timestamp,
                        "data_completion_status": "In Progress",
                        "last_updated": datetime.now().isoformat()
                    },
                    "collected_data": {},
                    "missing_fields": list(self.student_data.__fields__.keys())
                }
            }

            # Write initial profile
            with open(self.profile_filename, 'w', encoding='utf-8') as f:
                json.dump(initial_profile, f, indent=4, ensure_ascii=False)

            print(f"✅ Profile file initialized: {self.profile_filename}")
            return True

        except Exception as e:
            print(f"❌ Error initializing profile file: {e}")
            return False

    def extract_information_with_llm(self, user_message):
        """
        Use LLM to extract structured student info from a user message.
        Updates the student_data object and saves the profile.
        """
        current_data = self.student_data.dict()

        extraction_prompt = f"""
        Extract student information from the message and return ONLY a valid JSON object.

        Current data: {json.dumps(current_data, default=str)}

        User message: "{user_message}"

        Extract any of these fields (only include if clearly mentioned):
        - Marks_10th: percentage (0-100)
        - Marks_12th: percentage (0-100)
        - JEE_Score: rank/score (positive integer)
        - Budget: amount in rupees (convert lakhs/crores: 5 lakhs = 500000)
        - Preferred_Location: city/state in India
        - Gender: Male/Female/Other
        - Target_Exam: JEE/NEET/etc
        - State_Board: CBSE/ICSE/State Board/etc
        - Category: General/OBC/SC/ST/EWS/etc
        - Extra_Curriculars: activities/achievements
        - Future_Goal: career aspirations

        Return ONLY valid JSON. If nothing found, return {{}}.

        Examples:
        "I got 85% in 10th" → {{"Marks_10th": 85}}
        "Budget is 5 lakhs" → {{"Budget": 500000}}
        "I'm a boy from Delhi" → {{"Gender": "Male", "Preferred_Location": "Delhi"}}
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": extraction_prompt}],
                temperature=0.1,
                max_tokens=300,
            )

            extracted_text = response.choices[0].message.content.strip()
            print(f"🔍 LLM Extraction Response: {extracted_text}")

            # Parse JSON from response
            try:
                json_match = re.search(r'\{.*\}', extracted_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    extracted_data = json.loads(json_str)

                    if extracted_data:  # Only update if we extracted something
                        # Update current data with new values
                        for key, value in extracted_data.items():
                            if value is not None and value != "":
                                current_data[key] = value

                        # Validate with Pydantic
                        self.student_data = StudentData(**current_data)
                        print(f"📊 Updated student data: {self.student_data.dict()}")

                        # Save profile after each update
                        self.save_profile_json()
                        return True

            except json.JSONDecodeError as e:
                print(f"❌ JSON decode error: {e}")
            return False

        except Exception as e:
            print(f"❌ Error in extraction: {e}")
            return False

    def check_data_completion(self):
        """
        Check if all required student fields have been collected.
        """
        data_dict = self.student_data.dict()
        missing_fields = [k for k, v in data_dict.items() if v is None]
        return len(missing_fields) == 0

    def get_missing_fields(self):
        """
        Return a list of missing student data fields.
        """
        data_dict = self.student_data.dict()
        return [k for k, v in data_dict.items() if v is None]

    def get_next_question(self):
        """
        Suggest the next question to ask based on missing fields.
        """
        missing = self.get_missing_fields()
        if not missing:
            return None

        question_map = {
            'Marks_10th': "Could you please share your 10th standard marks percentage?",
            'Marks_12th': "What were your 12th standard marks percentage?",
            'JEE_Score': "Did you take JEE? If so, what was your score or rank?",
            'Budget': "What's your budget for the entire course (you can mention in lakhs)?",
            'Preferred_Location': "Which part of India would you prefer to study in?",
            'Gender': "Just for better recommendations, could you let me know your gender?",
            'Target_Exam': "Which entrance exams are you preparing for or have taken?",
            'State_Board': "Which educational board did you study under (CBSE, ICSE, State Board)?",
            'Category': "What's your reservation category (General, OBC, SC, ST, etc.)?",
            'Extra_Curriculars': "Do you have any extracurricular activities or achievements?",
            'Future_Goal': "What are your career goals or what field interests you?"
        }

        return question_map.get(missing[0], f"Could you tell me about your {missing[0]}?")

    def generate_comprehensive_profile_json(self):
        """
        Generate a comprehensive student profile (including recommendations if complete).
        """
        data_dict = self.student_data.dict()
        timestamp = datetime.now()

        # Get collected vs missing data
        collected_data = {k: v for k, v in data_dict.items() if v is not None}
        missing_fields = [k for k, v in data_dict.items() if v is None]

        profile_data = {
            "student_profile": {
                "session_info": {
                    "profile_created": timestamp.isoformat(),
                    "counselor_name": self.name,
                    "session_id": timestamp.strftime("%Y%m%d_%H%M%S"),
                    "data_completion_status": "Complete" if self.data_collected else "In Progress",
                    "missing_fields": missing_fields,
                    "collected_fields": list(collected_data.keys()),
                    "completion_percentage": f"{((11 - len(missing_fields)) / 11) * 100:.1f}%",
                    "last_updated": timestamp.isoformat()
                },
                "personal_information": {
                    "gender": data_dict.get('Gender'),
                    "category": data_dict.get('Category'),
                    "preferred_location": data_dict.get('Preferred_Location')
                },
                "academic_details": {
                    "marks_10th_percentage": data_dict.get('Marks_10th'),
                    "marks_12th_percentage": data_dict.get('Marks_12th'),
                    "jee_score_rank": data_dict.get('JEE_Score'),
                    "state_board": data_dict.get('State_Board'),
                    "target_examinations": data_dict.get('Target_Exam')
                },
                "preferences_and_goals": {
                    "education_budget_inr": data_dict.get('Budget'),
                    "career_goals": data_dict.get('Future_Goal'),
                    "extracurricular_activities": data_dict.get('Extra_Curriculars')
                },
                "raw_collected_data": collected_data
            }
        }

        # Add college recommendations if data is complete
        if self.data_collected:
            recommendations = self.get_college_recommendations()
            profile_data["college_recommendations"] = {
                "recommendation_date": timestamp.isoformat(),
                "total_recommendations": len(recommendations),
                "colleges": recommendations
            }

        return profile_data

    def get_college_recommendations(self):
        """
        Return a list of recommended colleges based on the student's profile.
        """
        recommendations = []
        data_dict = self.student_data.dict()

        for college in self.colleges:
            match_score = 0
            match_reasons = []

            # Location matching
            if (data_dict.get('Preferred_Location') and
                data_dict['Preferred_Location'].lower() in college['location'].lower()):
                match_score += 2
                match_reasons.append(f"Located in preferred area: {college['location']}")

            # Budget matching
            if data_dict.get('Budget') and college['fees'] <= data_dict['Budget']:
                match_score += 2
                match_reasons.append(f"Fits within budget (₹{college['fees']:,})")

            # JEE Score matching
            if (data_dict.get('JEE_Score') and college.get('min_jee') and
                data_dict['JEE_Score'] <= college['min_jee']):
                match_score += 3
                match_reasons.append(f"JEE rank qualifies (Min required: {college['min_jee']})")

            # Add basic matches for all colleges
            if match_score == 0:
                match_score = 1
                match_reasons.append("General recommendation based on profile")

            recommendations.append({
                "college_name": college['name'],
                "location": college['location'],
                "annual_fees_inr": college['fees'],
                "specialties": college['specialties'],
                "acceptance_rate": college['acceptance_rate'],
                "match_score": match_score,
                "match_reasons": match_reasons,
                "minimum_jee_rank_required": college.get('min_jee')
            })

        # Sort by match score and return top 5
        recommendations.sort(key=lambda x: x['match_score'], reverse=True)
        return recommendations[:5]

    def save_profile_json(self):
        """
        Save the current student profile to a JSON file.
        """
        try:
            profile_data = self.generate_comprehensive_profile_json()

            # Ensure we have a filename
            if not self.profile_filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.profile_filename = self.profiles_dir / f"student_profile_{timestamp}.txt"

            # Create directory if it doesn't exist
            self.profile_filename.parent.mkdir(parents=True, exist_ok=True)

            # Write with error handling
            with open(self.profile_filename, 'w', encoding='utf-8') as f:
                json.dump(profile_data, f, indent=4, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())

            print(f"✅ Profile saved to: {self.profile_filename}")
            return str(self.profile_filename)

        except Exception as e:
            print(f"❌ Error saving profile: {e}")
            return None

    def get_profile_content_for_download(self):
        """
        Return the current profile as a JSON string for download.
        """
        try:
            profile_data = self.generate_comprehensive_profile_json()
            return json.dumps(profile_data, indent=4, ensure_ascii=False, sort_keys=True)
        except Exception as e:
            print(f"❌ Error generating profile content: {e}")
            return json.dumps({"error": f"Could not generate profile: {str(e)}"}, indent=4)

    def chat(self, message, history):
        """
        Main chat handler: processes a user message, updates state, and generates a response.
        """
        print(f"💬 Processing message: {message[:50]}...")

        # Extract information from user message
        extraction_success = self.extract_information_with_llm(message)

        # Add user message to conversation history
        self.conversation_history.append({"role": "user", "content": message})

        # Check if we have all required information
        if not self.data_collected:
            self.data_collected = self.check_data_completion()
            if self.data_collected:
                print("🎉 All required data collected!")
                self.save_profile_json()  # Save complete profile

                student_profile = "\n".join([f"{k}: {v}" for k, v in self.student_data.dict().items()])
                colleges_data = self.format_colleges_for_prompt()

                self.conversation_history.append({
                    "role": "system",
                    "content": f"""
                    PROVIDE RECOMMENDATIONS NOW. All required information has been collected.

                    Student Profile:
                    {student_profile}

                    {colleges_data}

                    Based on this student's profile, recommend 3-5 suitable colleges that match their profile.
                    For each recommendation, explain:
                    1. Why this college is a good fit
                    2. Key programs relevant to their interests
                    3. Admission requirements and competitiveness
                    4. Estimated costs and how it fits their budget

                    Also provide practical next steps for applications.
                    End by mentioning they can download their complete profile using the download button.
                    """
                })

                self.recommendations_provided = True

            else:
                # Continue collecting information - save partial profile
                self.save_profile_json()  # Save after every update

                missing_fields = self.get_missing_fields()
                next_question = self.get_next_question()

                collected_info = {k: v for k, v in self.student_data.dict().items() if v is not None}

                self.conversation_history.append({
                    "role": "system",
                    "content": f"""
                    Continue collecting information. Acknowledge what they shared and ask: "{next_question}"

                    Information collected so far: {json.dumps(collected_info, default=str)}
                    Still missing: {missing_fields}

                    Be encouraging and conversational. Mention that their profile is being saved automatically.
                    """
                })

        try:
            # Generate response
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.conversation_history[-10:],  # Keep recent context
                temperature=0.7,
                max_tokens=1000,
            )

            assistant_response = response.choices[0].message.content

        except Exception as e:
            assistant_response = f"I'm sorry, there was an error: {str(e)}"
            print(f"❌ Chat Error: {e}")

        # Add response to history
        self.conversation_history.append({"role": "assistant", "content": assistant_response})

        return assistant_response

    def format_colleges_for_prompt(self):
        """
        Format the college database for inclusion in LLM prompts.
        """
        colleges_text = "College Database:\n"
        for college in self.colleges:
            colleges_text += f"- {college['name']}: Location: {college['location']}, "
            colleges_text += f"Min JEE Rank (if applicable): {college['min_jee']}, "
            colleges_text += f"Approximate Fees: {college['fees']}, "
            colleges_text += f"Acceptance Rate: {college['acceptance_rate']}, "
            colleges_text += f"Specialties: {', '.join(college['specialties'])}\n"
        return colleges_text

# =============================
# FastAPI Application & Endpoints
# =============================

# In-memory session store for active chatbot sessions (not persistent)
sessions = {}

# Create FastAPI app
app = FastAPI(title="College Counselor Chatbot API")

# Specify the port number
port = os.getenv("PORT", 8000)

# Enable CORS for all origins (for local dev/testing)
app.add_middleware( 
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI!"}

# Main entrypoint
if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))  # Use Render's provided $PORT
    uvicorn.run("main:app", host="0.0.0.0", port=port)

# Request model for initializing a session
class InitSessionRequest(BaseModel):
    name: Optional[str] = "User"  # Counselor name (default: Lauren)

@app.post("/init_session")
def init_session(req: InitSessionRequest):
    """
    Initialize a new chatbot session. Returns a session_id and profile file path.
    """
    try:
        api_key = os.getenv("API_KEY")
        counselor = CollegeCounselorChatbot(api_key, name=req.name)
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        sessions[session_id] = counselor
        return {"session_id": session_id, "profile_file": str(counselor.profile_filename)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize session: {e}")

# Request model for chat endpoint
class MessageRequest(BaseModel):
    session_id: str  # Session ID from /init_session
    message: str     # User's message to the chatbot
    history: Optional[list] = None  # Optional conversation history

@app.post("/chat")
def chat(req: MessageRequest):
    """
    Send a message to the chatbot and get a response. Requires a valid session_id.
    Returns the assistant's response, current profile status, and profile file path.
    """
    if req.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    counselor = sessions[req.session_id]
    try:
        response = counselor.chat(req.message, req.history or [])
        return {"response": response, "profile_status": counselor.student_data.dict(), "profile_file": str(counselor.profile_filename)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {e}")

@app.get("/profile/{session_id}")
def get_profile(session_id: str):
    """
    Get the current student profile as JSON for a given session.
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    counselor = sessions[session_id]
    try:
        content = counselor.get_profile_content_for_download()
        return JSONResponse(content=json.loads(content))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Profile error: {e}")

@app.get("/download_profile/{session_id}")
def download_profile(session_id: str):
    """
    Download the current student profile as a text file for a given session.
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    counselor = sessions[session_id]
    if not counselor.profile_filename or not os.path.exists(counselor.profile_filename):
        raise HTTPException(status_code=404, detail="Profile file not found")
    return FileResponse(str(counselor.profile_filename), filename=os.path.basename(counselor.profile_filename), media_type='text/plain')


