document.addEventListener("DOMContentLoaded", () => {
    const queryBox = document.getElementById("query-box");
    const sendBtn = document.querySelector(".send");
    const chatContainer = document.querySelector(".container");

    async function checkServerHealth() {
        try {
            const response = await fetch("http://localhost:8000/api/query/health");
            const data = await response.json();
            if (!data.ready) {
                return "Server is not ready. Please try again later.";
            }
            return null;
        } catch (error) {
            console.error("Error checking server health:", error);
            return "Error connecting to the server.";
        }
    }

    async function sendMessage() {
        const message = queryBox.value.trim();
        if (!message) return;

        // Add user message to chat
        const userDiv = document.createElement("div");
        userDiv.className = "user-message";
        userDiv.textContent = message;
        chatContainer.appendChild(userDiv);

        // Clear input field
        queryBox.value = "";

        // Add loading message
        const botDiv = document.createElement("div");
        botDiv.className = "bot-message";
        botDiv.textContent = "Thinking...";
        chatContainer.appendChild(botDiv);

        // Scroll to bottom
        setTimeout(() => {
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }, 100);

        // Check if server is ready
        const serverStatus = await checkServerHealth();
        if (serverStatus) {
            botDiv.textContent = serverStatus;
            return;
        }

        try {
            const response = await fetch("http://localhost:8000/api/query", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ query: message }),
            });

            const data = await response.json();

            // Update bot response
            botDiv.textContent = data.answer || "No response from the server.";

            // Show sources if available
            if (data.sources && data.sources.length > 0) {
                const sourcesDiv = document.createElement("div");
                sourcesDiv.className = "sources";
                sourcesDiv.textContent = `Sources: ${data.sources.join(", ")}`;
                chatContainer.appendChild(sourcesDiv);
            }
        } catch (error) {
            console.error("Error:", error);
            botDiv.textContent = "Error getting response from the server.";
        }

        // Scroll to bottom after response
        setTimeout(() => {
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }, 100);
    }

    if (sendBtn) sendBtn.addEventListener("click", sendMessage);
    if (queryBox) {
        queryBox.addEventListener("keypress", (e) => {
            if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
    }
});
