// DOM Elements
const chatMessages = document.getElementById('chat-messages');
const userInput = document.getElementById('user-input');
const sendButton = document.getElementById('send-button');
const quantumCoherence = document.getElementById('quantum-coherence');
const processingMode = document.getElementById('processing-mode');
const confidence = document.getElementById('confidence');
const processingTime = document.getElementById('processing-time');

// State
let isProcessing = false;

// Event Listeners
userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// Functions
async function sendMessage() {
    if (isProcessing) return;
    
    const message = userInput.value.trim();
    if (!message) return;
    
    try {
        // Set processing state
        isProcessing = true;
        sendButton.disabled = true;
        sendButton.classList.add('processing');
        
        // Add user message to chat immediately
        addMessage(message, 'user');
        
        // Clear input
        userInput.value = '';
        
        // Send message to server
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            body: JSON.stringify({ message: message })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        console.log('Received response:', data); // Debug log
        
        if (data && data.response) {
            // Add assistant's response to chat
            addMessage(data.response.text, 'assistant');
            
            // Update metrics
            updateMetrics(data.response);
        } else {
            throw new Error('Invalid response format');
        }
        
    } catch (error) {
        console.error('Error:', error);
        addMessage('Sorry, I encountered an error processing your request. Please try again.', 'assistant');
    } finally {
        // Reset processing state
        isProcessing = false;
        sendButton.disabled = false;
        sendButton.classList.remove('processing');
    }
}

function addMessage(text, sender) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', `${sender}-message`);
    messageDiv.textContent = text;
    
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function updateMetrics(response) {
    if (!response) return;
    
    // Update quantum coherence
    if (response.metadata && response.metadata.quantum_coherence !== undefined) {
        quantumCoherence.textContent = response.metadata.quantum_coherence.toFixed(3);
    }
    
    // Update processing mode
    if (response.metadata && response.metadata.processing_mode) {
        processingMode.textContent = response.metadata.processing_mode;
    }
    
    // Update confidence
    if (response.confidence !== undefined) {
        confidence.textContent = response.confidence.toFixed(3);
    }
    
    // Update processing time
    if (response.processing_time !== undefined) {
        processingTime.textContent = `${response.processing_time.toFixed(3)}s`;
    }
}

// Initialize typing animation for the placeholder
function initTypingAnimation() {
    const placeholders = [
        "Ask me anything...",
        "How can I assist you today?",
        "Let's explore quantum consciousness together...",
        "Type your message here..."
    ];
    let currentIndex = 0;
    
    setInterval(() => {
        userInput.placeholder = placeholders[currentIndex];
        currentIndex = (currentIndex + 1) % placeholders.length;
    }, 3000);
}

// Initialize the interface
initTypingAnimation();
