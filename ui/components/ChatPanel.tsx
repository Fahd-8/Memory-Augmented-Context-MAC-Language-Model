'use client'

import { useState, useRef, useEffect } from 'react'
import { Message } from '@/types'
import MessageBubble from './MessageBubble'
import ResetButton from './ResetButton'

interface Props {
  messages: Message[]
  onSend: (prompt: string) => void
  onReset: () => void
  loading: boolean
}

export default function ChatPanel({ messages, onSend, onReset, loading }: Props) {
  const [input, setInput] = useState('')
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const handleSend = () => {
    if (!input.trim() || loading) return
    onSend(input.trim())
    setInput('')
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <div style={{
        display: 'flex',
        alignItems: 'flex-start',
        justifyContent: 'space-between',
        marginBottom: 20
      }}>
        <div>
          <div style={{
            fontFamily: 'IBM Plex Mono',
            fontSize: 12,
            color: 'var(--text-2)',
            letterSpacing: '0.1em',
            marginBottom: 4
          }}>MAC CHAT</div>
          <p style={{ color: 'var(--text-3)', fontSize: 12 }}>
            LMM accumulates memory across messages
          </p>
        </div>
        <ResetButton onReset={onReset} loading={loading} />
      </div>

      <div style={{ flex: 1, overflowY: 'auto', paddingRight: 4, marginBottom: 16 }}>
        {messages.length === 0 && (
          <div style={{
            height: '100%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}>
            <div style={{ textAlign: 'center' }}>
              <div style={{
                fontFamily: 'IBM Plex Mono',
                fontSize: 11,
                color: 'var(--text-2)',
                lineHeight: 2
              }}>
                <div>› Tell the model your name</div>
                <div>› Tell it where you're from</div>
                <div>› Tell it what you love</div>
                <div style={{ marginTop: 16 }}>Then prove it remembers</div>
              </div>
            </div>
          </div>
        )}

        {messages.map(msg => (
          <MessageBubble key={msg.id} message={msg} />
        ))}

        {loading && (
          <div style={{ display: 'flex', justifyContent: 'flex-start', marginBottom: 12 }}>
            <div style={{
              marginRight: 12,
              width: 20,
              height: 20,
              background: 'var(--bg-3)',
              border: '1px solid var(--border-light)',
              borderRadius: 4,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontFamily: 'IBM Plex Mono',
              fontSize: 9,
              color: 'var(--text-3)',
              flexShrink: 0,
              marginTop: 2
            }}>M</div>
            <div style={{
              borderLeft: '1px solid var(--border-light)',
              paddingLeft: 14,
              paddingTop: 2
            }}>
              <span style={{
                fontFamily: 'IBM Plex Mono',
                fontSize: 13,
                color: 'var(--text-3)',
                animation: 'blink 1s infinite'
              }}>processing_</span>
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      <div style={{
        display: 'flex',
        gap: 8,
        borderTop: '3px solid var(--border)',
        paddingTop: 16
      }}>
        <div style={{
          flex: 1,
          display: 'flex',
          alignItems: 'center',
          background: 'var(--bg-2)',
          border: '3px solid var(--border)',
          borderRadius: 6,
          padding: '0 12px',
          gap: 8
        }}>
          <span style={{
            fontFamily: 'IBM Plex Mono',
            fontSize: 12,
            color: 'var(--text-3)',
            flexShrink: 0
          }}>›</span>
          <textarea
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Input message..."
            rows={1}
            style={{
              flex: 1,
              background: 'transparent',
              border: 'none',
              outline: 'none',
              color: 'var(--text)',
              fontFamily: 'IBM Plex Mono',
              fontSize: 13,
              resize: 'none',
              padding: '10px 0',
              lineHeight: 2.3
            }}
          />
        </div>
        <button
          onClick={handleSend}
          disabled={loading || !input.trim()}
          style={{
            width: 40,
            height: 40,
            background: input.trim() && !loading ? 'var(--bg-3)' : 'transparent',
            border: '2px solid var(--border)',
            borderRadius: 6,
            color: input.trim() && !loading ? 'var(--text)' : 'var(--text-3)',
            cursor: loading || !input.trim() ? 'not-allowed' : 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            transition: 'all 0.15s ease',
            flexShrink: 0,
            alignSelf: 'flex-end'
          }}
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="m22 2-7 20-4-9-9-4Z"/>
            <path d="M22 2 11 13"/>
          </svg>
        </button>
      </div>
    </div>
  )
}