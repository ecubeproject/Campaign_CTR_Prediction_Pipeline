# Must match (or exceed) the Python version used to train/pickle the model
# -- scikit-learn 1.9.0 requires Python >=3.11, and loading a pickle with a
# different sklearn version than pickled it can silently produce wrong
# results rather than an error. See requirements.txt for the pinned version.
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy everything
COPY . .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose port used by Streamlit
EXPOSE 8080

CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
