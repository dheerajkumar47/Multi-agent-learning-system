# app.py - Updated with Vision Agent
import streamlit as st
import sys
import os
import numpy as np
from PIL import Image
import cv2

# Add the current directory to Python path so we can import our agents
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.base_agent import BaseAgent
from agents.vision_agent import VisionAgent

def main():
    st.set_page_config(
        page_title="Multi-Agent System",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Multi-Agent Learning System")
    st.markdown("**Day 2**: Vision Agent Integration")
    
    # Initialize agents in session state
    if 'agents' not in st.session_state:
        with st.spinner("Loading AI models..."):
            # In app.py, inside the session_state setup:
           st.session_state.agents = {
             'vision': VisionAgent(emotion_model_path="models/emotion_model.h5"),
             'alice': BaseAgent("Alice", ["communication", "echo"]),
             'bob': BaseAgent("Bob", ["communication", "echo"]),
            'charlie': BaseAgent("Charlie", ["communication", "echo"])
}
    
    # Sidebar: Agent status
    st.sidebar.header("🤖 Agent Status")
    
    for name, agent in st.session_state.agents.items():
        if name == 'vision':
            status = agent.get_status()
            # Check if vision models loaded
            status_msg = agent.process_message({"type": "status"})
            if status_msg.get('status') == 'success':
                agent_status = status_msg.get('agent_status', {})
                if agent_status.get('ready', False):
                    st.sidebar.success(f"🔍 Vision Agent: Online")
                    st.sidebar.write(f"Emotion Model: {'✅' if agent_status.get('emotion_model_loaded') else '❌'}")
                    st.sidebar.write(f"Object Model: {'✅' if agent_status.get('object_model_loaded') else '❌'}")
                else:
                    st.sidebar.warning(f"🔍 Vision Agent: Loading...")
            else:
                st.sidebar.error(f"🔍 Vision Agent: Error")
        else:
            status = agent.get_status()
            st.sidebar.success(f"✅ {status['name']}: Online")
            st.sidebar.write(f"Messages: {status['message_count']}")
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Vision Analysis", "🗣️ Agent Communication", "📊 Performance"])
    
    with tab1:
        st.header("🔍 Vision Agent Analysis")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Upload Image")
            
            uploaded_file = st.file_uploader(
                "Choose an image file", 
                type=['jpg', 'jpeg', 'png'],
                help="Upload an image for emotion and object detection"
            )
            
            analysis_type = st.selectbox(
                "Analysis Type:",
                ["Complete Analysis", "Emotions Only", "Objects Only"]
            )
            
            if uploaded_file is not None:
                # Display the image
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                if st.button("🔍 Analyze Image", type="primary"):
                    with st.spinner("Vision Agent analyzing image..."):
                        
                        # Prepare message for vision agent
                        if analysis_type == "Complete Analysis":
                            message = {
                                "type": "analyze_image",
                                "image_data": image,
                                "sender": "user"
                            }
                        elif analysis_type == "Emotions Only":
                            message = {
                                "type": "detect_emotions_only",
                                "image_data": image,
                                "sender": "user"
                            }
                        else:  # Objects Only
                            message = {
                                "type": "detect_objects_only",
                                "image_data": image,
                                "sender": "user"
                            }
                        
                        # Send to vision agent
                        response = st.session_state.agents['vision'].process_message(message)
                        
                        # Store result in session state
                        if 'vision_results' not in st.session_state:
                            st.session_state.vision_results = []
                        
                        st.session_state.vision_results.append({
                            "image": image,
                            "analysis_type": analysis_type,
                            "response": response,
                            "timestamp": st.session_state.agents['vision']._get_timestamp()
                        })
                        
                        st.success("✅ Analysis complete!")
        
        with col2:
            st.subheader("Analysis Results")
            
            if 'vision_results' in st.session_state and st.session_state.vision_results:
                latest_result = st.session_state.vision_results[-1]
                response = latest_result['response']
                
                if response.get('status') == 'success':
                    
                    # Display Complete Analysis
                    if 'analysis' in response:
                        analysis = response['analysis']['scene_analysis']
                        
                        # Scene Description
                        if 'description' in analysis:
                            st.subheader("🎯 Scene Description")
                            scene_desc = analysis['description']
                            st.write(scene_desc['natural_language'])
                            
                            # Summary metrics
                            summary = scene_desc['summary']
                            col_a, col_b, col_c = st.columns(3)
                            with col_a:
                                st.metric("People", summary['people_count'])
                            with col_b:
                                st.metric("Objects", summary['object_count'])
                            with col_c:
                                st.metric("Main Emotion", summary['dominant_emotion'])
                        
                        # Detailed Emotions
                        if 'emotions' in analysis and analysis['emotions'].get('emotions'):
                            st.subheader("😊 Emotion Analysis")
                            for i, emotion_data in enumerate(analysis['emotions']['emotions']):
                                with st.expander(f"Person {i+1}: {emotion_data['dominant_emotion']} (Confidence: {emotion_data['confidence']:.2f})"):
                                    # Show all emotion probabilities
                                    emotions = emotion_data['all_emotions']
                                    for emotion, prob in sorted(emotions.items(), key=lambda x: x[1], reverse=True):
                                        st.write(f"{emotion}: {prob:.3f}")
                        
                        # Detailed Objects
                        if 'objects' in analysis and analysis['objects'].get('objects'):
                            st.subheader("📦 Object Detection")
                            objects = analysis['objects']['objects'][:10]  # Show top 10
                            for obj in objects:
                                st.write(f"• {obj['object']} (confidence: {obj['confidence']:.2f})")
                    
                    # Display Emotion-only results
                    elif 'emotions' in response:
                        st.subheader("😊 Emotion Analysis Results")
                        emotions = response['emotions']
                        if emotions.get('emotions'):
                            for i, emotion_data in enumerate(emotions['emotions']):
                                st.write(f"**Person {i+1}:** {emotion_data['dominant_emotion']} (Confidence: {emotion_data['confidence']:.2f})")
                        else:
                            st.info(emotions.get('message', 'No emotions detected'))
                    
                    # Display Object-only results
                    elif 'objects' in response:
                        st.subheader("📦 Object Detection Results")
                        objects = response['objects']
                        if objects.get('objects'):
                            for obj in objects['objects'][:10]:
                                st.write(f"• {obj['object']} (confidence: {obj['confidence']:.2f})")
                        else:
                            st.info(objects.get('message', 'No objects detected'))
                
                else:
                    st.error(f"Analysis failed: {response.get('error', 'Unknown error')}")
                
                # Show analysis history
                if len(st.session_state.vision_results) > 1:
                    st.subheader("📚 Analysis History")
                    for i, result in enumerate(reversed(st.session_state.vision_results[:-1])):
                        with st.expander(f"Analysis {len(st.session_state.vision_results)-i-1}: {result['analysis_type']}"):
                            st.write(f"Timestamp: {result['timestamp']}")
                            # Show brief summary
                            if result['response'].get('status') == 'success':
                                st.write("Status: ✅ Success")
                            else:
                                st.write(f"Status: ❌ {result['response'].get('error', 'Failed')}")
            
            else:
                st.info("Upload an image and click 'Analyze' to see results here!")
    
    with tab2:
        st.header("🗣️ Agent Communication Test")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Send Message")
            
            sender = st.selectbox("From Agent:", list(st.session_state.agents.keys()))
            receiver = st.selectbox("To Agent:", [k for k in st.session_state.agents.keys() if k != sender])
            
            message_type = st.selectbox("Message Type:", ["greeting", "question", "command", "test", "status"])
            message_content = st.text_area("Message Content:", "Hello! How are you?")
            
            if st.button("Send Message 📤"):
                if message_content.strip():
                    # Prepare message
                    message = {
                        "type": message_type,
                        "content": message_content,
                        "sender": sender
                    }
                    
                    # Send message
                    sender_agent = st.session_state.agents[sender]
                    receiver_agent = st.session_state.agents[receiver]
                    
                    with st.spinner(f"Sending message from {sender} to {receiver}..."):
                        response = sender_agent.send_message_to(receiver_agent, message)
                    
                    # Store in session state for display
                    if 'conversation' not in st.session_state:
                        st.session_state.conversation = []
                    
                    st.session_state.conversation.append({
                        "from": sender,
                        "to": receiver,
                        "message": message_content,
                        "response": response.get('response', str(response)),
                        "status": response.get('status', 'unknown')
                    })
                    
                    st.success("Message sent!")
                else:
                    st.error("Please enter a message!")
        
        with col2:
            st.subheader("Conversation History")
            
            if 'conversation' in st.session_state and st.session_state.conversation:
                for i, conv in enumerate(reversed(st.session_state.conversation[-10:])):  # Show last 10
                    with st.expander(f"💬 {conv['from']} → {conv['to']}", expanded=(i==0)):
                        st.write(f"**Message:** {conv['message']}")
                        st.write(f"**Response:** {conv['response']}")
                        st.write(f"**Status:** {conv['status']}")
            else:
                st.info("No conversations yet. Send a message to get started!")
    
    with tab3:
        st.header("📊 Agent Performance Metrics")
        
        # Vision Agent specific metrics
        st.subheader("🔍 Vision Agent Performance")
        if 'vision_results' in st.session_state and st.session_state.vision_results:
            vision_results = st.session_state.vision_results
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Analyses", len(vision_results))
            
            with col2:
                successful = sum(1 for r in vision_results if r['response'].get('status') == 'success')
                st.metric("Successful Analyses", successful)
            
            with col3:
                success_rate = (successful / len(vision_results)) * 100 if vision_results else 0
                st.metric("Success Rate", f"{success_rate:.1f}%")
            
            # Analysis type breakdown
            st.subheader("Analysis Types")
            analysis_types = {}
            for result in vision_results:
                atype = result['analysis_type']
                analysis_types[atype] = analysis_types.get(atype, 0) + 1
            
            for atype, count in analysis_types.items():
                st.write(f"• {atype}: {count}")
        
        # General agent performance
        st.subheader("🤖 All Agents Performance")
        
        perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
        
        for i, (name, agent) in enumerate(st.session_state.agents.items()):
            col = [perf_col1, perf_col2, perf_col3, perf_col4][i % 4]
            with col:
                status = agent.get_status()
                st.metric(
                    label=f"{name.title()} Messages",
                    value=status['message_count']
                )
                if name != 'vision':  # Vision agent doesn't track response time the same way
                    st.metric(
                        label=f"{name.title()} Avg Response",
                        value=f"{status['performance']['average_response_time']:.3f}s"
                    )
    
    # Clear button
    st.sidebar.markdown("---")
    if st.sidebar.button("🗑️ Clear All History"):
        # Clear agent histories
         for agent in st.session_state.agents.values():
          agent.clear_history()
    
        # Clear session state variables
         if 'conversation' in st.session_state:
          st.session_state.conversation = []
         if 'vision_results' in st.session_state:
          st.session_state.vision_results = []
    
         st.sidebar.success("History cleared!")
         st.rerun()  # ✅ Use st.rerun() instead of experimental_rerun

if __name__ == "__main__":
    main()