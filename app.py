import streamlit as st
from support import academic_research_component, classroom_overview_component, personalized_learning_component, automatic_grading_component, student_report_component, settings_component

# CSS style for the sidebar
sidebar_style = """
.sidebar .block-container {
    display: flex;
    flex-direction: column;
    justify-content: space-between; /* This makes the buttons uniformly distributed */
}

.sidebar .block-container .block {
    margin-bottom: 1rem;
    width: 100%;
}

.stButton>button {
    width: 100%;
}


"""

# CSS for the settings button
settings_style = """
.settings-btn {
    cursor: pointer;
    font-size: 1.5rem;
    margin-left: 0.5rem;
}
"""


def main():
    st.sidebar.title("Educator's Menu")

    # Apply the custom CSS style
    st.markdown(f'<style>{sidebar_style}{settings_style}</style>', unsafe_allow_html=True)




    
    # Initialize session states if they don't exist
    components = [
        'academic_research', 
        'classroom_overview', 
        'personalized_learning', 
        'automatic_grading', 
        'student_report', 
        'settings'
    ]
    for component in components:
        if component not in st.session_state:
            st.session_state[component] = False

    # Update the session states when the buttons are clicked
    if st.sidebar.button("Academic Research"):
        st.session_state.academic_research = True
        for component in components:
            if component != 'academic_research':
                st.session_state[component] = False

    if st.sidebar.button("Classroom Overview"):
        st.session_state.classroom_overview = True
        for component in components:
            if component != 'classroom_overview':
                st.session_state[component] = False

    if st.sidebar.button("Personalized Learning"):
        st.session_state.personalized_learning = True
        for component in components:
            if component != 'personalized_learning':
                st.session_state[component] = False

    if st.sidebar.button("Automatic Grading"):
        st.session_state.automatic_grading = True
        for component in components:
            if component != 'automatic_grading':
                st.session_state[component] = False

    if st.sidebar.button("Student Report"):
        st.session_state.student_report = True
        for component in components:
            if component != 'student_report':
                st.session_state[component] = False
    if st.sidebar.button("Settings"):
        st.session_state.settings = True
        for component in components:
            if component != 'settings':
                st.session_state[component] = False

    # Only call the respective component if its session state is True
    if st.session_state.academic_research:
        academic_research_component()
    if st.session_state.classroom_overview:
        classroom_overview_component()
    if st.session_state.personalized_learning:
        personalized_learning_component()
    if st.session_state.automatic_grading:
        automatic_grading_component()
    if st.session_state.student_report:
        student_report_component()
    if st.session_state.settings:
        settings_component()



if __name__ == '__main__':
    main()