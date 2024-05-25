import './App.css';
import React from 'react';
import '@chatui/core/es/styles/index.less';
import './chatui-theme.css';
import Chat, { Bubble, useMessages } from '@chatui/core';
import '@chatui/core/dist/index.css';
import OpenAI from 'openai';

const openai = new OpenAI({
  apiKey: '0000',
  dangerouslyAllowBrowser: true,
  baseURL: "https://u401435-9b28-df6d5176.westb.seetacloud.com:8443/v1"
});
var message_history = [];

function App() {
  const { messages, appendMsg, setTyping, updateMsg } = useMessages([]);

  async function chat_stream(prompt, _msgId) {
    message_history.push({ role: 'user', content: prompt });
    const stream = openai.beta.chat.completions.stream({
      model: 'ChatGLM3-6B',
      messages: message_history,
      stream: true,
    });
    var full_text = "";
    for await (const chunk of stream) {
      if (chunk.choices[0]?.delta?.content === undefined) {
        continue;
      }
      full_text = full_text + chunk.choices[0]?.delta?.content || '';
      updateMsg(_msgId, {
        type: "text",
        content: { text: full_text.trim() }
      });
    }
    message_history.push({ "role": "assistant", "content": full_text });
  }

  function handleSend(type, val) {
    if (type === 'text' && val.trim()) {
      appendMsg({
        type: 'text',
        content: { text: val },
        position: 'right',
      });
      setTyping(true);
      const msgID = new Date().getTime();
      appendMsg({
        _id: msgID,
        type: 'text',
        content: { text: '' },
      });
      chat_stream(val, msgID);
    }
  }

  function renderMessageContent(msg) {
    const { content } = msg;
    return <Bubble content={content.text} />;
  }

  return (
    <Chat
      navbar={{ title: 'chat-app' }}
      messages={messages}
      renderMessageContent={renderMessageContent}
      onSend={handleSend}
    />
  );
}

export default App;
