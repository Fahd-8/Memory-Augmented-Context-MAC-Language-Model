import { Message } from '@/types'

interface Props {
  message: Message
}

export default function MessageBubble({ message }: Props) {
  const isUser = message.role === 'user'

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-3 fade-up`}>
      {!isUser && (
        <div className="mr-3 mt-1 flex-shrink-0">
          <div style={{
            width: 20,
            height: 20,
            background: 'var(--bg-3)',
            border: '2px solid var(--border-light)',
            borderRadius: 4,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontFamily: 'IBM Plex Mono',
            fontSize: 9,
            color: 'var(--text-2)',
            letterSpacing: 0
          }}>M</div>
        </div>
      )}
      <div style={{
        maxWidth: '72%',
        padding: isUser ? '8px 14px' : '10px 14px',
        background: isUser ? 'var(--bg-3)' : 'transparent',
        border: isUser ? '1px solid var(--border)' : 'none',
        borderRadius: isUser ? 8 : 0,
        borderLeft: !isUser ? '1px solid var(--border-light)' : undefined,
        paddingLeft: !isUser ? 14 : undefined,
      }}>
        {!isUser && (
          <div style={{
            fontFamily: 'IBM Plex Mono',
            fontSize: 10,
            color: 'var(--text-2)',
            marginBottom: 4,
            letterSpacing: '0.05em'
          }}>MAC_OUTPUT</div>
        )}
        <p style={{
          fontFamily: isUser ? 'IBM Plex Sans' : 'IBM Plex Mono',
          fontSize: isUser ? 14 : 13,
          color: isUser ? 'var(--text)' : 'var(--text)',
          lineHeight: 1.7,
          whiteSpace: 'pre-wrap'
        }}>{message.content}</p>
      </div>
    </div>
  )
}