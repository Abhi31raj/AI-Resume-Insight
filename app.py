from flask import Flask, request, jsonify, render_template, redirect, url_for,flash
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_cors import CORS
from flask import session
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
from pdf2image import convert_from_path
import pytesseract
import pdfplumber
import google.generativeai as genai
import os
import json
import random
import re
import time  # Import time for sleep function
import traceback
import subprocess
import tempfile
import difflib

# --- your app setup ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise Exception("GOOGLE_API_KEY not set in .env file")

genai.configure(api_key=GOOGLE_API_KEY)

app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


UPLOAD_DIR = 'uploads'
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ---------- Text Extraction from PDF ----------
def get_resume_text(pdf_file_path):
    try:
        text = ""
        with pdfplumber.open(pdf_file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
        if text.strip():
            return text.strip()
    except Exception as e:
        print(f"pdfplumber failed: {e}")

    try:
        images = convert_from_path(pdf_file_path)
        text = ""
        for image in images:
            text += pytesseract.image_to_string(image)
        return text.strip()
    except Exception as e:
        print(f"OCR failed: {e}")
        raise

# ---------- Extract Candidate Name ----------
def find_candidate_name(resume_text):
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"Extract the full name of the candidate from this resume text:\n{resume_text}\nReturn only the name."
    try:
        response = model.generate_content(prompt)
        return response.text.strip().split('\n')[0]
    except:
        return "Unknown"

# ---------- Calculate ATS Score ----------
def compute_ats_score(text_input, jd_input):
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"""
You are an AI trained to simulate ATS systems.
Give a score from 0 to 100 based on:
- Skill match
- Experience relevance
- Certifications
- Formatting

Resume:
{text_input}

Job Description:
{jd_input}

Respond only with the score (e.g., 82.5)
"""
    try:
        response = model.generate_content(prompt)
        match = re.search(r'\d{1,3}(?:\.\d+)?', response.text)
        return float(match.group(0)) if match else 0.0
    except:
        return 0.0

# ---------- Resume Feedback ----------
def resume_recommendations(text_input, jd_input=None):
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"""
You are a resume improvement assistant. Analyze the resume below:

1. Extract and list key skills and certifications.
2. Give constructive feedback.
3. Recommend helpful courses.
4. Justify your recommendations.

Resume:
{text_input}
"""
    if jd_input:
        prompt += f"\n\nJob Description:\n{jd_input}"

    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error generating feedback: {e}"

# ---------- Resume Quality Breakdown ----------
def detailed_resume_scores(text_input):
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"""
Provide flat JSON scores out of 100 for these resume aspects:
- Skills
- Work Experience
- Certifications
- Formatting

Resume:
{text_input}

Format:
{{"Skills": 80, "Work Experience": 75, "Certifications": 60, "Formatting": 85}}
Only return JSON.
"""
    try:
        response = model.generate_content(prompt)
        match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if match:
            return eval(match.group())
    except:
        pass

    return {
        "Skills": 50,
        "Work Experience": 50,
        "Certifications": 50,
        "Formatting": 50
    }


@app.route('/analyze12', methods=['POST'])
def analyze_resume():
    if 'resume' not in request.files:
        return jsonify({'error': 'No resume file provided'}), 400

    file = request.files['resume']
    job_desc = request.form.get('jobDescription', '')

    if file.filename == '':
        return jsonify({'error': 'Empty file uploaded'}), 400

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    file.save(file_path)

    try:
        resume_text = get_resume_text(file_path)
        if not resume_text:
            return jsonify({'error': 'Unable to extract text from resume'}), 500

        name = find_candidate_name(resume_text)
        ats_score = compute_ats_score(resume_text, job_desc)
        feedback = resume_recommendations(resume_text, job_desc)

        before_improve = detailed_resume_scores(resume_text)
        after_improve = {
            k: min((v if isinstance(v, (int, float)) else 0) + 20, 100)
            for k, v in before_improve.items()
        }

        return jsonify({
            'name': name,
            'ats_score': ats_score,
            'feedback': feedback,
            'before': before_improve,
            'after': after_improve
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f"Analysis failed: {str(e)}"}), 500



# Configure SQLite Database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = "your_secret_key"

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

# User Model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), nullable=False, unique=True)
    email = db.Column(db.String(120), nullable=False, unique=True)
    password = db.Column(db.String(128), nullable=False)

# Initialize the database
with app.app_context():
    db.create_all()

def safe_generate_content(prompt, retries=3, delay=2):
    """Safely call Gemini API with retries."""
    model = genai.GenerativeModel("gemini-1.5-flash")
    for attempt in range(retries):
        try:
            response = model.generate_content(prompt)
            if response and response.text:
                return response.text.strip()
        except Exception as e:
            print(f"[Gemini Retry {attempt+1}] {e}")
            time.sleep(delay)
    raise Exception("Failed to get a valid response from Gemini after retries.")

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"[pdfplumber] Failed: {e}")

    if text.strip():
        return text.strip()

    try:
        images = convert_from_path(pdf_path)
        for img in images:
            text += pytesseract.image_to_string(img)
    except Exception as e:
        print(f"[OCR] Failed: {e}")

    if text.strip():
        return text.strip()

    try:
        import fitz
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text()
    except Exception as e:
        print(f"[PyMuPDF] Failed: {e}")

    if text.strip():
        return text.strip()

    raise Exception("Failed to extract text from PDF.")

def extract_name(text):
    prompt = f"""Extract ONLY the full name of the candidate from the following resume.
If not found, respond exactly as "Unknown".
Resume Text:
{text}

Respond ONLY with the name."""
    try:
        response_text = safe_generate_content(prompt)
        response_text = response_text.replace('\n', ' ').strip()
        name_match = re.search(r'\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)+)\b', response_text)
        if name_match:
            return name_match.group(1)
        # fallback: try first non-empty line
        first_line = next((line for line in text.splitlines() if line.strip()), None)
        if first_line:
            words = first_line.strip().split()
            if len(words) >= 2:
                return f"{words[0]} {words[1]}"
        return "Unknown"
    except Exception as e:
        print(f"[Name Extraction] Failed: {e}")
        return "Unknown"

def calculate_ats_score(resume_text, job_description):
    prompt = f"""You are an ATS system. Match the following resume to the job description.

Resume:
{resume_text}

Job Description:
{job_description}

Respond ONLY with a score between 0 to 100 (no text, no extra lines)."""
    try:
        response_text = safe_generate_content(prompt)
        match = re.search(r'\d{1,3}(?:\.\d+)?', response_text)
        return float(match.group(0)) if match else 0.0
    except Exception as e:
        print(f"[ATS Scoring] Failed: {e}")
        return 0.0

def analyze_resume(resume_text, job_description=None):
    prompt = f"""You are a professional resume reviewer. Analyze the following resume:

1. List current skills and certifications.
2. Suggest improvements.
3. Recommend 2-3 useful courses.
4. Justify each suggestion.

Resume:
{resume_text}
"""
    if job_description:
        prompt += f"\nAlso compare with this Job Description:\n{job_description}"

    try:
        return safe_generate_content(prompt)
    except Exception as e:
        print(f"[Resume Feedback] Failed: {e}")
        return "Feedback generation failed."




@app.route('/aianalysis2')
def aianalysis2():
    return render_template('aianalysis2.html')

    
@app.route('/aianalysis1')
def aianalysis1():
    return render_template('aianalysis1.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    global latest_ats_score, latest_resume_text

    if 'resume' not in request.files:
        return jsonify({'error': 'No resume file provided.'}), 400

    file = request.files['resume']
    if file.filename == '':
        return jsonify({'error': 'Empty file uploaded.'}), 400

    job_description = request.form.get('jobDescription', '')

    try:
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
    except Exception as e:
        return jsonify({'error': f'Failed to save resume: {e}'}), 500

    results = {
        'name': 'Unknown',
        'ats_score': 0.0,
        'feedback': 'No feedback generated.'
    }
    error_messages = []

    try:
        resume_text = extract_text_from_pdf(file_path)
        if not resume_text:
            return jsonify({'error': 'Unable to extract text from resume.'}), 500
        latest_resume_text = resume_text
    except Exception as e:
        return jsonify({'error': f"Failed to extract text: {e}"}), 500

    try:
        results['name'] = extract_name(resume_text)
    except Exception as e:
        error_messages.append(f"Name extraction failed: {e}")

    try:
        ats_score = calculate_ats_score(resume_text, job_description)
        results['ats_score'] = ats_score
        latest_ats_score = ats_score
    except Exception as e:
        error_messages.append(f"ATS scoring failed: {e}")

    try:
        results['feedback'] = analyze_resume(resume_text, job_description)
    except Exception as e:
        error_messages.append(f"Resume feedback generation failed: {e}")

    response = {
        'status': 'success',
        'data': results
    }

    if error_messages:
        response['warnings'] = error_messages

    return jsonify(response), 200



def get_starter_code(language):
    return {
        "Python": "def main():\n    # Write your code here\n    pass",
        "JavaScript": "function main() {\n    // Write your code here\n}",
        "Java": "public class Main {\n    public static void main(String[] args) {\n        // Write your code here\n    }\n}",
        "C++": "#include <iostream>\nusing namespace std;\n\nint main() {\n    // Write your code here\n    return 0;\n}"
    }.get(language, "")

latest_ats_score = None
latest_resume_text = None

reference_solutions = {
    "Previous question example": {
        "Python": "print('This is a previous question')"
    },  

    "Longest subarray with sum 0": {
        "Python": """def longest_subarray_with_sum_zero(arr):
    sum_to_index = {}
    max_len = 0
    current_sum = 0
    for i in range(len(arr)):
        current_sum += arr[i]
        if current_sum == 0:
            max_len = i + 1
        elif current_sum in sum_to_index:
            max_len = max(max_len, i - sum_to_index[current_sum])
        else:
            sum_to_index[current_sum] = i
    return max_len

arr = [15, -2, 2, -8, 1, 7, 10, 23]
print("Length of longest subarray with sum 0:", longest_subarray_with_sum_zero(arr))""",

        "Java": """import java.util.*;
public class LongestZeroSumSubarray {
    public static int longestSubarray(int[] arr) {
        Map<Integer, Integer> map = new HashMap<>();
        int sum = 0, maxLen = 0;
        for (int i = 0; i < arr.length; i++) {
            sum += arr[i];
            if (sum == 0) maxLen = i + 1;
            else if (map.containsKey(sum)) maxLen = Math.max(maxLen, i - map.get(sum));
            else map.put(sum, i);
        }
        return maxLen;
    }
    public static void main(String[] args) {
        int[] arr = {15, -2, 2, -8, 1, 7, 10, 23};
        System.out.println("Length: " + longestSubarray(arr));
    }
}""",

        "JavaScript": """function longestSubarrayWithZeroSum(arr) {
    const map = new Map();
    let sum = 0, maxLen = 0;
    for (let i = 0; i < arr.length; i++) {
        sum += arr[i];
        if (sum === 0) maxLen = i + 1;
        else if (map.has(sum)) maxLen = Math.max(maxLen, i - map.get(sum));
        else map.set(sum, i);
    }
    return maxLen;
}
const arr = [15, -2, 2, -8, 1, 7, 10, 23];
console.log("Length:", longestSubarrayWithZeroSum(arr));""",

        "C++": """#include <iostream>
#include <unordered_map>
using namespace std;

int longestSubarrayWithZeroSum(int arr[], int n) {
    unordered_map<int, int> map;
    int sum = 0, maxLen = 0;
    for (int i = 0; i < n; i++) {
        sum += arr[i];
        if (sum == 0) maxLen = i + 1;
        else if (map.count(sum)) maxLen = max(maxLen, i - map[sum]);
        else map[sum] = i;
    }
    return maxLen;
}

int main() {
    int arr[] = {15, -2, 2, -8, 1, 7, 10, 23};
    int n = sizeof(arr)/sizeof(arr[0]);
    cout << "Length: " << longestSubarrayWithZeroSum(arr, n);
    return 0;
}"""
    },  # <-- This closes the "Longest subarray with sum 0" dictionary

    "Find the smallest missing positive number in an unsorted array.arr = [3, 4, -1, 1] ": {
  "Python": """def first_missing_positive(nums):
    n = len(nums)
    for i in range(n):
        while 1 <= nums[i] <= n and nums[nums[i] - 1] != nums[i]:
            nums[nums[i] - 1], nums[i] = nums[i], nums[nums[i] - 1]
    for i in range(n):
        if nums[i] != i + 1:
            return i + 1
    return n + 1

arr = [3, 4, -1, 1]
print("First missing positive:", first_missing_positive(arr))""",
  "Java": """public class FirstMissingPositive {
    public static int firstMissingPositive(int[] nums) {
        int n = nums.length;
        for (int i = 0; i < n; i++) {
            while (nums[i] > 0 && nums[i] <= n && nums[nums[i] - 1] != nums[i]) {
                int temp = nums[nums[i] - 1];
                nums[nums[i] - 1] = nums[i];
                nums[i] = temp;
            }
        }
        for (int i = 0; i < n; i++) {
            if (nums[i] != i + 1) return i + 1;
        }
        return n + 1;
    }

    public static void main(String[] args) {
        int[] arr = {3, 4, -1, 1};
        System.out.println("First missing positive: " + firstMissingPositive(arr));
    }
}""",
  "JavaScript": """function firstMissingPositive(nums) {
    const n = nums.length;
    for (let i = 0; i < n; i++) {
        while (nums[i] > 0 && nums[i] <= n && nums[nums[i] - 1] !== nums[i]) {
            let temp = nums[nums[i] - 1];
            nums[nums[i] - 1] = nums[i];
            nums[i] = temp;
        }
    }
    for (let i = 0; i < n; i++) {
        if (nums[i] !== i + 1) return i + 1;
    }
    return n + 1;
}
console.log("First missing positive:", firstMissingPositive([3, 4, -1, 1]));""",
  "C++": """#include <iostream>
using namespace std;

int firstMissingPositive(int arr[], int n) {
    for (int i = 0; i < n; i++) {
        while (arr[i] > 0 && arr[i] <= n && arr[arr[i] - 1] != arr[i]) {
            swap(arr[i], arr[arr[i] - 1]);
        }
    }
    for (int i = 0; i < n; i++) {
        if (arr[i] !== i + 1)
            return i + 1;
    }
    return n + 1;
}

int main() {
    int arr[] = {3, 4, -1, 1};
    int n = sizeof(arr)/sizeof(arr[0]);
    cout << "First missing positive: " << firstMissingPositive(arr, n);
    return 0;
}"""
    },

"Find the length of the longest sequence of consecutive integers. arr = {100, 4, 200, 1, 3, 2}": {
  "Python": """def longest_consecutive(nums):
    num_set = set(nums)
    longest = 0
    for num in num_set:
        if num - 1 not in num_set:
            current = num
            streak = 1
            while current + 1 in num_set:
                current += 1
                streak += 1
            longest = max(longest, streak)
    return longest

arr = [100, 4, 200, 1, 3, 2]
print("Longest consecutive sequence:", longest_consecutive(arr))""",
  "Java": """import java.util.*;

public class LongestConsecutive {
    public static int longestConsecutive(int[] nums) {
        Set<Integer> set = new HashSet<>();
        for (int num : nums) set.add(num);
        int maxLen = 0;

        for (int num : set) {
            if (!set.contains(num - 1)) {
                int curr = num, len = 1;
                while (set.contains(curr + 1)) {
                    curr++;
                    len++;
                }
                maxLen = Math.max(maxLen, len);
            }
        }
        return maxLen;
    }

    public static void main(String[] args) {
        int[] arr = {100, 4, 200, 1, 3, 2};
        System.out.println("Longest consecutive sequence: " + longestConsecutive(arr));
    }
}""",
  "JavaScript": """function longestConsecutive(nums) {
    const set = new Set(nums);
    let maxLen = 0;

    for (let num of set) {
        if (!set.has(num - 1)) {
            let curr = num, len = 1;
            while (set.has(curr + 1)) {
                curr++;
                len++;
            }
            maxLen = Math.max(maxLen, len);
        }
    }
    return maxLen;
}
console.log("Longest consecutive sequence:", longestConsecutive([100, 4, 200, 1, 3, 2]));""",
  "C++": """#include <iostream>
#include <unordered_set>
using namespace std;

int longestConsecutive(int arr[], int n) {
    unordered_set<int> set(arr, arr + n);
    int maxLen = 0;

    for (int num : set) {
        if (set.find(num - 1) == set.end()) {
            int curr = num, len = 1;
            while (set.find(curr + 1) != set.end()) {
                curr++;
                len++;
            }
            maxLen = max(maxLen, len);
        }
    }
    return maxLen;
}

int main() {
    int arr[] = {100, 4, 200, 1, 3, 2};
    int n = sizeof(arr) / sizeof(arr[0]);
    cout << "Longest consecutive sequence: " << longestConsecutive(arr, n);
    return 0;
}"""
    },

"Maximum Subarray (Kadane s Algorithm) arr = [-2,1,-3,4,-1,2,1,-5,4]": {
  "Python": """def max_subarray(nums):
    max_ending = max_so_far = nums[0]
    for num in nums[1:]:
        max_ending = max(num, max_ending + num)
        max_so_far = max(max_so_far, max_ending)
    return max_so_far

arr = [-2,1,-3,4,-1,2,1,-5,4]
print("Maximum subarray sum:", max_subarray(arr))""",
  "Java": """public class MaxSubarray {
    public static int maxSubArray(int[] nums) {
        int maxEnding = nums[0], maxSoFar = nums[0];
        for (int i = 1; i < nums.length; i++) {
            maxEnding = Math.max(nums[i], maxEnding + nums[i]);
            maxSoFar = Math.max(maxSoFar, maxEnding);
        }
        return maxSoFar;
    }

    public static void main(String[] args) {
        int[] arr = {-2,1,-3,4,-1,2,1,-5,4};
        System.out.println("Maximum subarray sum: " + maxSubArray(arr));
    }
}""",
  "JavaScript": """function maxSubArray(nums) {
    let maxEnding = nums[0];
    let maxSoFar = nums[0];
    for (let i = 1; i < nums.length; i++) {
        maxEnding = Math.max(nums[i], maxEnding + nums[i]);
        maxSoFar = Math.max(maxSoFar, maxEnding);
    }
    return maxSoFar;
}

const arr = [-2,1,-3,4,-1,2,1,-5,4];
console.log("Maximum subarray sum:", maxSubArray(arr));""",
  "C++": """#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

int maxSubArray(vector<int>& nums) {
    int maxEnding = nums[0], maxSoFar = nums[0];
    for (int i = 1; i < nums.size(); i++) {
        maxEnding = max(nums[i], maxEnding + nums[i]);
        maxSoFar = max(maxSoFar, maxEnding);
    }
    return maxSoFar;
}

int main() {
    vector<int> arr = {-2,1,-3,4,-1,2,1,-5,4};
    cout << "Maximum subarray sum: " << maxSubArray(arr) << endl;
    return 0;
}"""
    },

"Product of Array Except Self . arr = [1,2,3,4]": {
  "Python": """def product_except_self(nums):
    length = len(nums)
    answer = [1] * length
    left = 1
    for i in range(length):
        answer[i] = left
        left *= nums[i]
    right = 1
    for i in range(length - 1, -1, -1):
        answer[i] *= right
        right *= nums[i]
    return answer

arr = [1,2,3,4]
print("Product of array except self:", product_except_self(arr)""",
  "Java": """public class ProductExceptSelf {
    public static int[] productExceptSelf(int[] nums) {
        int length = nums.length;
        int[] answer = new int[length];
        answer[0] = 1;

        for (int i = 1; i < length; i++) {
            answer[i] = nums[i - 1] * answer[i - 1];
        }

        int right = 1;
        for (int i = length - 1; i >= 0; i--) {
            answer[i] = answer[i] * right;
            right *= nums[i];
        }

        return answer;
    }

    public static void main(String[] args) {
        int[] arr = {1,2,3,4};
        int[] result = productExceptSelf(arr);
        System.out.print("Product of array except self: [");
        for (int i = 0; i < result.length; i++) {
            System.out.print(result[i]);
            if (i != result.length - 1) System.out.print(", ");
        }
        System.out.println("]");
    }
}""",
  "JavaScript": """function productExceptSelf(nums) {
    const length = nums.length;
    const answer = new Array(length).fill(1);
    let left = 1;

    for (let i = 0; i < length; i++) {
        answer[i] = left;
        left *= nums[i];
    }

    let right = 1;
    for (let i = length - 1; i >= 0; i--) {
        answer[i] *= right;
        right *= nums[i];
    }
    return answer;
}

const arr = [1,2,3,4];
console.log("Product of array except self:", productExceptSelf(arr));""",
  "C++": """#include <iostream>
#include <vector>
using namespace std;

vector<int> productExceptSelf(vector<int>& nums) {
    int length = nums.size();
    vector<int> answer(length, 1);
    int left = 1;
    for (int i = 0; i < length; i++) {
        answer[i] = left;
        left *= nums[i];
    }
    int right = 1;
    for (int i = length - 1; i >= 0; i--) {
        answer[i] *= right;
        right *= nums[i];
    }
    return answer;
}

int main() {
    vector<int> arr = {1,2,3,4};
    vector<int> result = productExceptSelf(arr);
    cout << "Product of array except self: [";
    for (int i = 0; i < result.size(); i++) {
        cout << result[i];
        if (i != result.size() - 1) cout << ", ";
    }
    cout << "]" << endl;
    return 0;
}"""
    },

"Longest Increasing Subsequence.  arr = [10,9,2,5,3,7,101,18]": {
  "Python": """def length_of_lis(nums):
    if not nums:
        return 0
    dp = [1] * len(nums)
    for i in range(len(nums)):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)

arr = [10,9,2,5,3,7,101,18]
print("Length of LIS:", length_of_lis(arr))""",
  "Java": """public class LongestIncreasingSubsequence {
    public static int lengthOfLIS(int[] nums) {
        if (nums.length == 0) return 0;
        int[] dp = new int[nums.length];
        Arrays.fill(dp, 1);
        int maxLen = 1;
        for (int i = 1; i < nums.length; i++) {
            for (int j = 0; j < i; j++) {
                if (nums[j] < nums[i]) {
                    dp[i] = Math.max(dp[i], dp[j] + 1);
                }
            }
            maxLen = Math.max(maxLen, dp[i]);
        }
        return maxLen;
    }

    public static void main(String[] args) {
        int[] arr = {10,9,2,5,3,7,101,18};
        System.out.println("Length of LIS: " + lengthOfLIS(arr));
    }
}""",
  "JavaScript": """function lengthOfLIS(nums) {
    if (!nums.length) return 0;
    const dp = new Array(nums.length).fill(1);
    let maxLen = 1;
    for (let i = 1; i < nums.length; i++) {
        for (let j = 0; j < i; j++) {
            if (nums[j] < nums[i]) {
                dp[i] = Math.max(dp[i], dp[j] + 1);
            }
        }
        maxLen = Math.max(maxLen, dp[i]);
    }
    return maxLen;
}

const arr = [10,9,2,5,3,7,101,18];
console.log("Length of LIS:", lengthOfLIS(arr));""",
  "C++": """#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

int lengthOfLIS(vector<int>& nums) {
    if (nums.empty()) return 0;
    vector<int> dp(nums.size(), 1);
    int maxLen = 1;
    for (int i = 1; i < nums.size(); i++) {
        for (int j = 0; j < i; j++) {
            if (nums[j] < nums[i]) {
                dp[i] = max(dp[i], dp[j] + 1);
            }
        }
        maxLen = max(maxLen, dp[i]);
    }
    return maxLen;
}

int main() {
    vector<int> arr = {10,9,2,5,3,7,101,18};
    cout << "Length of LIS: " << lengthOfLIS(arr) << endl;
    return 0;
}"""
    },

"Median of Two Sorted Arrays.  nums1 = [1,3]  and nums2 = [2]": {
  "Python": """def find_median_sorted_arrays(nums1, nums2):
    A, B = nums1, nums2
    total = len(A) + len(B)
    half = total // 2
    if len(A) > len(B):
        A, B = B, A
    left, right = 0, len(A)
    while left <= right:
        i = (left + right) // 2
        j = half - i

        Aleft = A[i - 1] if i > 0 else float('-inf')
        Aright = A[i] if i < len(A) else float('inf')
        Bleft = B[j - 1] if j > 0 else float('-inf')
        Bright = B[j] if j < len(B) else float('inf')

        if Aleft <= Bright and Bleft <= Aright:
            if total % 2:
                return min(Aright, Bright)
            return (max(Aleft, Bleft) + min(Aright, Bright)) / 2
        elif Aleft > Bright:
            right = i - 1
        else:
            left = i + 1

nums1 = [1,3]
nums2 = [2]
print("Median is:", find_median_sorted_arrays(nums1, nums2))""",
  "Java": """public class MedianSortedArrays {
    public static double findMedianSortedArrays(int[] nums1, int[] nums2) {
        if (nums1.length > nums2.length) return findMedianSortedArrays(nums2, nums1);

        int x = nums1.length;
        int y = nums2.length;
        int low = 0, high = x;

        while (low <= high) {
            int partitionX = (low + high) / 2;
            int partitionY = (x + y + 1) / 2 - partitionX;

            int maxX = (partitionX == 0) ? Integer.MIN_VALUE : nums1[partitionX - 1];
            int maxY = (partitionY == 0) ? Integer.MIN_VALUE : nums2[partitionY - 1];

            int minX = (partitionX == x) ? Integer.MAX_VALUE : nums1[partitionX];
            int minY = (partitionY == y) ? Integer.MAX_VALUE : nums2[partitionY];

            if (maxX <= minY && maxY <= minX) {
                if ((x + y) % 2 == 0) {
                    return (double)(Math.max(maxX, maxY) + Math.min(minX, minY)) / 2;
                } else {
                    return (double)Math.max(maxX, maxY);
                }
            } else if (maxX > minY) {
                high = partitionX - 1;
            } else {
                low = partitionX + 1;
            }
        }
        throw new IllegalArgumentException();
    }

    public static void main(String[] args) {
        int[] nums1 = {1,3};
        int[] nums2 = {2};
        System.out.println("Median is: " + findMedianSortedArrays(nums1, nums2));
    }
}""",
  "JavaScript": """function findMedianSortedArrays(nums1, nums2) {
    if (nums1.length > nums2.length) return findMedianSortedArrays(nums2, nums1);
    let x = nums1.length, y = nums2.length;
    let low = 0, high = x;

    while (low <= high) {
        let partitionX = Math.floor((low + high) / 2);
        let partitionY = Math.floor((x + y + 1) / 2) - partitionX;

        let maxX = partitionX === 0 ? -Infinity : nums1[partitionX - 1];
        let maxY = partitionY === 0 ? -Infinity : nums2[partitionY - 1];

        let minX = partitionX === x ? Infinity : nums1[partitionX];
        let minY = partitionY === y ? Infinity : nums2[partitionY];

        if (maxX <= minY && maxY <= minX) {
            if ((x + y) % 2 === 0) {
                return (Math.max(maxX, maxY) + Math.min(minX, minY)) / 2;
            } else {
                return Math.max(maxX, maxY);
            }
        } else if (maxX > minY) {
            high = partitionX - 1;
        } else {
            low = partitionX + 1;
        }
    }
    throw new Error("Input arrays are not sorted");
}

const nums1 = [1,3];
const nums2 = [2];
console.log("Median is:", findMedianSortedArrays(nums1, nums2));""",
  "C++": """#include <iostream>
#include <vector>
#include <climits>
using namespace std;

double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
    if (nums1.size() > nums2.size()) return findMedianSortedArrays(nums2, nums1);

    int x = nums1.size();
    int y = nums2.size();
    int low = 0, high = x;

    while (low <= high) {
        int partitionX = (low + high) / 2;
        int partitionY = (x + y + 1) / 2 - partitionX;

        int maxX = (partitionX == 0) ? INT_MIN : nums1[partitionX - 1];
        int maxY = (partitionY == 0) ? INT_MIN : nums2[partitionY - 1];

        int minX = (partitionX == x) ? INT_MAX : nums1[partitionX];
        int minY = (partitionY == y) ? INT_MAX : nums2[partitionY];

        if (maxX <= minY && maxY <= minX) {
            if ((x + y) % 2 == 0) {
                return (double)(max(maxX, maxY) + min(minX, minY)) / 2;
            } else {
                return (double)max(maxX, maxY);
            }
        } else if (maxX > minY) {
            high = partitionX - 1;
        } else {
            low = partitionX + 1;
        }
    }
    throw invalid_argument("Input arrays are not sorted");
}

int main() {
    vector<int> nums1 = {1,3};
    vector<int> nums2 = {2};
    cout << "Median is: " << findMedianSortedArrays(nums1, nums2) << endl;
    return 0;
}"""
    },

"Trapping Rain Water. height = [0,1,0,2,1,0,1,3,2,1,2,1]": {
  "Python": """def trap(height):
    if not height:
        return 0
    left, right = 0, len(height) - 1
    left_max, right_max = 0, 0
    trapped = 0

    while left < right:
        if height[left] < height[right]:
            if height[left] >= left_max:
                left_max = height[left]
            else:
                trapped += left_max - height[left]
            left += 1
        else:
            if height[right] >= right_max:
                right_max = height[right]
            else:
                trapped += right_max - height[right]
            right -= 1
    return trapped

height = [0,1,0,2,1,0,1,3,2,1,2,1]
print("Trapped rain water:", trap(height))""",
  "Java": """public class TrappingRainWater {
    public static int trap(int[] height) {
        if (height == null || height.length == 0) return 0;
        int left = 0, right = height.length - 1;
        int leftMax = 0, rightMax = 0;
        int trapped = 0;

        while (left < right) {
            if (height[left] < height[right]) {
                if (height[left] >= leftMax) leftMax = height[left];
                else trapped += leftMax - height[left];
                left++;
            } else {
                if (height[right] >= rightMax) rightMax = height[right];
                else trapped += rightMax - height[right];
                right--;
            }
        }
        return trapped;
    }

    public static void main(String[] args) {
        int[] height = {0,1,0,2,1,0,1,3,2,1,2,1};
        System.out.println("Trapped rain water: " + trap(height));
    }
}""",
  "JavaScript": """function trap(height) {
    if (!height.length) return 0;
    let left = 0, right = height.length - 1;
    let leftMax = 0, rightMax = 0;
    let trapped = 0;

    while (left < right) {
        if (height[left] < height[right]) {
            if (height[left] >= leftMax) leftMax = height[left];
            else trapped += leftMax - height[left];
            left++;
        } else {
            if (height[right] >= rightMax) rightMax = height[right];
            else trapped += rightMax - height[right];
            right--;
        }
    }
    return trapped;
}

const height = [0,1,0,2,1,0,1,3,2,1,2,1];
console.log("Trapped rain water:", trap(height));""",
  "C++": """#include <iostream>
#include <vector>
using namespace std;

int trap(vector<int>& height) {
    if (height.empty()) return 0;
    int left = 0, right = height.size() - 1;
    int leftMax = 0, rightMax = 0;
    int trapped = 0;

    while (left < right) {
        if (height[left] < height[right]) {
            if (height[left] >= leftMax) leftMax = height[left];
            else trapped += leftMax - height[left];
            left++;
        } else {
            if (height[right] >= rightMax) rightMax = height[right];
            else trapped += rightMax - height[right];
            right--;
        }
    }
    return trapped;
}

int main() {
    vector<int> height = {0,1,0,2,1,0,1,3,2,1,2,1};
    cout << "Trapped rain water: " << trap(height) << endl;
    return 0;
}"""
    },

"Longest Substring Without Repeating Characters. (abcabcbb)": {
  "Python": """def length_of_longest_substring(s):
    char_index = {}
    start = max_len = 0

    for i, ch in enumerate(s):
        if ch in char_index and char_index[ch] >= start:
            start = char_index[ch] + 1
        char_index[ch] = i
        max_len = max(max_len, i - start + 1)
    return max_len

print(length_of_longest_substring("abcabcbb"))  # Output: 3""",
  "Java": """import java.util.*;

public class LongestSubstring {
    public static int lengthOfLongestSubstring(String s) {
        Map<Character, Integer> map = new HashMap<>();
        int start = 0, maxLen = 0;

        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (map.containsKey(c) && map.get(c) >= start) {
                start = map.get(c) + 1;
            }
            map.put(c, i);
            maxLen = Math.max(maxLen, i - start + 1);
        }
        return maxLen;
    }

    public static void main(String[] args) {
        System.out.println(lengthOfLongestSubstring("abcabcbb"));  // 3
    }
}""",
  "JavaScript": """function lengthOfLongestSubstring(s) {
    const map = new Map();
    let start = 0, maxLen = 0;

    for (let i = 0; i < s.length; i++) {
        const c = s[i];
        if (map.has(c) && map.get(c) >= start) {
            start = map.get(c) + 1;
        }
        map.set(c, i);
        maxLen = Math.max(maxLen, i - start + 1);
    }
    return maxLen;
}

console.log(lengthOfLongestSubstring("abcabcbb")); // 3""",
  "C++": """#include <iostream>
#include <unordered_map>
#include <string>
using namespace std;

int lengthOfLongestSubstring(string s) {
    unordered_map<char,int> map;
    int start = 0, maxLen = 0;

    for (int i = 0; i < s.size(); i++) {
        char c = s[i];
        if (map.count(c) && map[c] >= start) {
            start = map[c] + 1;
        }
        map[c] = i;
        maxLen = max(maxLen, i - start + 1);
    }
    return maxLen;
}

int main() {
    cout << lengthOfLongestSubstring("abcabcbb") << endl; // 3
    return 0;
}"""
    },

"Find the Duplicate Number (Floyd's Tortoise and Hare cycle detection) .  [1,3,4,2,2]": {
  "Python": """def find_duplicate(nums):
    slow = fast = nums[0]
    while True:
        slow = nums[slow]
        fast = nums[nums[fast]]
        if slow == fast:
            break
    slow = nums[0]
    while slow != fast:
        slow = nums[slow]
        fast = nums[fast]
    return slow

print(find_duplicate([1,3,4,2,2]))  # Output: 2""",
  "Java": """public class FindDuplicate {
    public static int findDuplicate(int[] nums) {
        int slow = nums[0], fast = nums[0];
        do {
            slow = nums[slow];
            fast = nums[nums[fast]];
        } while (slow != fast);

        slow = nums[0];
        while (slow != fast) {
            slow = nums[slow];
            fast = nums[fast];
        }
        return slow;
    }

    public static void main(String[] args) {
        int[] nums = {1,3,4,2,2};
        System.out.println(findDuplicate(nums));  // 2
    }
}""",
  "JavaScript": """function findDuplicate(nums) {
    let slow = nums[0], fast = nums[0];
    do {
        slow = nums[slow];
        fast = nums[nums[fast]];
    } while (slow !== fast);

    slow = nums[0];
    while (slow !== fast) {
        slow = nums[slow];
        fast = nums[fast];
    }
    return slow;
}

console.log(findDuplicate([1,3,4,2,2])); // 2""",
  "C++": """#include <iostream>
#include <vector>
using namespace std;

int findDuplicate(vector<int>& nums) {
    int slow = nums[0], fast = nums[0];
    do {
        slow = nums[slow];
        fast = nums[nums[fast]];
    } while (slow != fast);

    slow = nums[0];
    while (slow != fast) {
        slow = nums[slow];
        fast = nums[fast];
    }
    return slow;
}

int main() {
    vector<int> nums = {1,3,4,2,2};
    cout << findDuplicate(nums) << endl; // 2
    return 0;
}"""
    },

    
}  # This closes the reference_solutions dictionary

def clean_lines(code_string):
    return [line.strip() for line in code_string.strip().splitlines() if line.strip()]

def compare_solutions(user_code, reference_code, matched_question):
    """Compare user's solution with reference solution and validate outputs."""
    try:
        # Define test cases
        test_cases = {
            "array except self": [
                ([1,2,3,4], [24,12,8,6]),
                ([0,0], [0,0])
            ],
            "longest consecutive": [
                ([100,4,200,1,3,2], 4),
                ([0,3,7,2,5,8,4,6,0,1], 9)
            ],
            "missing positive": [
                ([3,4,-1,1], 2),
                ([1,2,0], 3)
            ],
            "maximum subarray": [
                ([-2,1,-3,4,-1,2,1,-5,4], 6),
                ([1], 1)
            ],
            "trapping rain water": [
                ([0,1,0,2,1,0,1,3,2,1,2,1], 6),
                ([4,2,0,3,2,5], 9)
            ],
            "longest substring": [
                ("abcabcbb", 3),
                ("bbbbb", 1)
            ],
            "duplicate number": [
                ([1,3,4,2,2], 2),
                ([3,1,3,4,2], 3)
            ]
        }

        # Find matching test cases
        matched_test_cases = None
        for key in test_cases:
            if key.lower() in matched_question.lower():
                matched_test_cases = test_cases[key]
                break

        if not matched_test_cases:
            return 0

        # Get function name
        func_name = extract_function_name(user_code)
        if not func_name:
            return 0

        # Create test file
        test_code = f'''
{user_code}

def test_solution():
    test_cases = {matched_test_cases}
    passed = 0
    total = len(test_cases)
    
    for i, (test_input, expected) in enumerate(test_cases):
        try:
            result = {func_name}(test_input)
            if isinstance(expected, (list, tuple)):
                if list(result) == list(expected):
                    passed += 1
            else:
                if result == expected:
                    passed += 1
        except Exception as e:
            print(f"Error on test {{i}}: {{str(e)}}")
            continue
    
    print(f"PASSED:{{passed}},TOTAL:{{total}}")

test_solution()
'''

        # Write test code to file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_code)
            temp_file = f.name

        try:
            # Run tests
            result = subprocess.run(['python', temp_file], 
                                 capture_output=True, 
                                 text=True, 
                                 timeout=5)

            # Check for execution errors
            if result.returncode != 0:
                return 0

            # Parse results
            output = result.stdout
            if "PASSED:" not in output:
                return 0

            # Extract passed and total tests
            passed_str = output[output.find("PASSED:") + 7:output.find(",TOTAL:")]
            total_str = output[output.find("TOTAL:") + 6:]
            
            try:
                passed = int(passed_str)
                total = int(total_str)
                
                # Calculate score
                if passed == total:
                    return 100
                elif passed > 0:
                    return 70
                return 0
            except:
                return 0

        except:
            return 0
        finally:
            try:
                os.unlink(temp_file)
            except:
                pass

    except:
        return 0

def evaluate_user_code(user_code, language):
    try:
        # Clean user code
        user_code = user_code.strip()
        if not user_code:
            return {
                "score": 0,
                "combined_score": 0,
                "tips": "No code submitted."
            }

        # Find matching question
        matched_question = None
        matched_solution = None

        # Extract function name from user code
        user_func_name = extract_function_name(user_code)
        
        for question, solutions in reference_solutions.items():
            if language not in solutions:
                continue
                
            correct_solution = solutions[language]
            correct_func_name = extract_function_name(correct_solution)
            
            if user_func_name and correct_func_name and (
                user_func_name.lower() == correct_func_name.lower() or
                user_func_name.lower() in correct_func_name.lower() or
                correct_func_name.lower() in user_func_name.lower()
            ):
                matched_question = question
                matched_solution = correct_solution
                break

        if not matched_question:
            return {
                "score": 0,
                "combined_score": 0,
                "tips": "Could not identify which problem you are trying to solve. Make sure your function name matches the problem requirements."
            }

        # Compare the solutions
        score = compare_solutions(user_code, matched_solution, matched_question)
        
        # Generate feedback based on score
        if score == 0:
            tips = "Your solution is incorrect. Make sure your code:\n" + \
                   "1. Produces the correct output for the test cases\n" + \
                   "2. Uses the required algorithmic approach\n" + \
                   "3. Includes all necessary programming constructs (loops, conditionals, etc.)\n" + \
                   "4. Has proper syntax and runs without errors"
        elif score >= 90:
            tips = "Excellent implementation! Your solution matches the expected approach and produces correct output."
        elif score >= 70:
            tips = "Good attempt! Your solution is structurally similar but might need optimization or better handling of edge cases."
        else:
            tips = "Your solution needs improvement. Make sure you're using the correct algorithmic approach and handling all cases."

        return {
            "score": score,
            "combined_score": score,
            "tips": tips,
            "matched_question": matched_question
        }

    except Exception as e:
        print(f"Evaluation error: {e}")
        return {
            "score": 0,
            "combined_score": 0,
            "tips": f"Error during evaluation: {str(e)}"
        }

def extract_function_name(code):
    """Extract the main function name from code."""
    try:
        patterns = [
            r'def\s+([a-zA-Z_]\w*)\s*\(',  # Python
            r'function\s+([a-zA-Z_]\w*)\s*\(',  # JavaScript
            r'public\s+(?:static\s+)?[a-zA-Z_]\w*\s+([a-zA-Z_]\w*)\s*\(',  # Java
            r'(?:public\s+)?[a-zA-Z_]\w*\s+([a-zA-Z_]\w*)\s*\('  # C++
        ]
        
        for pattern in patterns:
            match = re.search(pattern, code)
            if match:
                return match.group(1)
        return None
    except:
        return None

@app.route('/evaluate_code', methods=['POST'])
def evaluate_code_endpoint():
    try:
        data = request.get_json()
        user_code = data.get('code', '')
        language = data.get('language', 'Python')

        result = evaluate_user_code(user_code, language)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ... rest of the code ...

@app.before_request
def before_request():
    if request.endpoint == 'splash':  # Check if it's the splash route
        time.sleep(3)  # Sleep for 3 seconds

@app.route('/')
def splash():
    # Render the splash page when the user accesses the root URL
    return render_template('splash.html')

@app.route('/login')
def login():
    # After the splash page, redirect to the login page
    return render_template('login.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')
    
# Route for the main page (index.html)
@app.route('/main')
def main_page():
    # Retrieve the username from the session
    username = session.get('username')

    # If no username is found in session, redirect to login (to handle unauthorized access)
    if not username:
        return redirect(url_for('login'))

    return render_template('index.html', username=username)  # Pass username to the template
  # Ensure `index.html` is in your `templates` folder.

@app.route('/home')
def home():
    return render_template('index.html')  # or any other template you want to render

@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/get-username')
def get_username():
    # Check if the username is stored in session
    username = session.get('username')

    if username:
        return jsonify({'username': username})
    else:
        return jsonify({'username': None})  # Return None if not logged in

@app.route('/signup', methods=['POST'])
def signup():
    data = request.get_json()
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')

    # Perform server-side validations and user creation
    if not username or not email or not password:
        return jsonify({"error": "All fields are required!"}), 400

    # Check if user already exists
    if User.query.filter_by(email=email).first():
        return jsonify({'error': 'Email already exists!'}), 400

    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
    new_user = User(username=username, email=email, password=hashed_password)

    db.session.add(new_user)
    db.session.commit()

    # Save the username in the session
    session['username'] = username

    return jsonify({'message': 'User registered successfully!', 'redirect_url': url_for('main_page')}), 201



SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587  # Use 465 for SSL, 587 for TLS
SENDER_EMAIL = "rvceclubnexus@gmail.com"
SENDER_PASSWORD = "idxwiyqgemmbxfam"  #vikaskt2005@gmail.com Use app password if you have 2-step verification enabled

# Function to send the email
def send_email(name, from_email, message):
    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = SENDER_EMAIL
    msg['Subject'] = f"Feedback from {name}"

    body = f"""
    Name: {name}
    Email: {from_email}
    Message: {message}
    """
    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.set_debuglevel(1)  # Enables debugging output to track the process
        server.starttls()  # Start TLS encryption
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.send_message(msg)
        server.quit()
        return True
    except smtplib.SMTPAuthenticationError as e:
        print(f"SMTP Authentication Error: {e}")
        return False
    except smtplib.SMTPException as e:
        print(f"SMTP Error: {e}")
        return False
    except Exception as e:
        print(f"General Error: {e}")
        return False

# Route for showing the feedback form
@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    if request.method == 'POST':
        # Get data from the form
        name = request.form.get('name')
        from_email = request.form.get('email')
        message = request.form.get('message')

        # Send email after form submission
        if send_email(name, from_email, message):
            flash('Thank you for your feedback!', 'success')
            return redirect('/thankyou')  # Redirect to a thank you page after successful submission
        else:
            flash('Error sending feedback. Please try again.', 'error')

    # Render the feedback form on GET request
    return render_template('feedback.html')

# Route for the thank you page
@app.route('/thankyou')
def thank_you():
    return render_template('thankyou.html')  # This is a simple thank you page

@app.route('/techai')
def AI_Assistant():
    return render_template('techai.html')
 


@app.route('/login', methods=['POST'])
def login_post():
    data = request.json
    email = data.get('email')
    password = data.get('password')

    if not (email and password):
        return jsonify({'error': 'Both fields are required!'}), 400

    user = User.query.filter_by(email=email).first()

    if not user or not bcrypt.check_password_hash(user.password, password):
        return jsonify({'error': 'Invalid credentials!'}), 401

    # Save the user's username in the session
    session['username'] = user.username

    # Redirect to the main page upon successful login
    return jsonify({'message': 'Login successful!', 'redirect_url': url_for('main_page')}), 200

@app.route('/logout')
def logout():
    session.pop('username', None)  # Remove the username from the session
    return redirect('splash.html')  # Redirect to the login page



@app.route('/guest-login')
def guest_login():
    # Clear the username from the session if it exists
    session.pop('username', None)
    return render_template('index.html')  # Render the index.html template

  # Redirect to the main page directly for guest login

NAV_ITEMS = [
    {"name": "Home", "url": "home", "icon": "fas fa-home"},
    {"name": "Categories", "url": "categories", "icon": "fas fa-list"},
    {"name": "Contact", "url": "contact", "icon": "fas fa-envelope"},
    {"name": "Logout", "url": "logout", "icon": "fas fa-sign-out-alt"},
    {"name": "AI Assistance", "url": "ai_assistance", "icon": "fas fa-robot"},
]

@app.route('/get_challenge')
def get_challenge():
    # Get a random question from reference_solutions
    questions = list(reference_solutions.keys())
    # Filter out the "Previous question example"
    questions = [q for q in questions if q != "Previous question example"]
    
    if not questions:
        return jsonify({'error': 'No questions available'}), 500
        
    random_question = random.choice(questions)
    
    return jsonify({
        'question': random_question,
        'languages': list(reference_solutions[random_question].keys())
    })

if __name__ == '__main__':
    app.run(debug=True)
