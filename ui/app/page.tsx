'use client'

import { useState, useEffect } from 'react'
import { Message, CompareResult } from '@/types'
import ChatPanel from '@/components/ChatPanel'
import MemoryProof from '@/components/MemoryProof'
import { generate, compare, reset, health } from '@/lib/api'

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([])
  const [loading, setLoading] = useState(false)
  const [compareLoading, setCompareLoading] = useState(false)
  const [compareResult, setCompareResult] = useState<CompareResult | null>(null)
  const [modelLoaded, setModelLoaded] = useState(false)
  const [device, setDevice] = useState('')

  useEffect(() => {
    health()
      .then(data => {
        setModelLoaded(data.model_loaded)
        setDevice(data.device)
      })
      .catch(() => setModelLoaded(false))
  }, [])

  const handleSend = async (prompt: string) => {
    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: prompt,
      timestamp: new Date()
    }
    setMessages(prev => [...prev, userMessage])
    setLoading(true)

    try {
      const response = await generate(prompt)
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: response,
        timestamp: new Date()
      }
      setMessages(prev => [...prev, assistantMessage])
    } catch {
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: 'ERR: cannot reach MAC API. ensure server is running on :8000',
        timestamp: new Date()
      }
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setLoading(false)
    }
  }

  const handleReset = async () => {
    try {
      await reset()
      setMessages([])
      setCompareResult(null)
    } catch (err) {
      console.error('Reset failed:', err)
    }
  }

  const handleProve = async () => {
    setCompareLoading(true)
    try {
      const result = await compare('What do you know about me so far?')
      setCompareResult({
        prompt: 'What do you know about me so far?',
        ttt_on: result.ttt_on,
        ttt_off: result.ttt_off
      })
    } catch (err) {
      console.error('Compare failed:', err)
    } finally {
      setCompareLoading(false)
    }
  }

  return (
    <main style={{ minHeight: '100vh', background: 'var(--bg)' }}>

      {/* top bar */}
      <div style={{
        borderBottom: '6px solid var(--border)',
        padding: '18px 28px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        background: 'var(--bg-2)'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
          <div style={{
            fontFamily: 'IBM Plex Mono',
            fontSize: 20,
            color: 'var(--text)',
            letterSpacing: '0.19em'
          }}>MAC</div>
          <div style={{
            width: 1,
            height: 14,
            background: 'var(--border-light)'
          }}/>
          <div style={{
            fontFamily: 'IBM Plex Mono',
            fontSize: 14,
            color: 'var(--text-3)',
            letterSpacing: '0.06em'
          }}>MEMORY-AUGMENTED CONTEXT · TITANS ARCHITECTURE</div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <div style={{
            width: 6,
            height: 6,
            borderRadius: '50%',
            background: modelLoaded ? 'var(--green)' : 'var(--red)',
            boxShadow: modelLoaded ? '0 0 6px rgba(74,222,128,0.4)' : '0 0 6px rgba(248,113,113,0.4)'
          }}/>
          <span style={{
            fontFamily: 'IBM Plex Mono',
            fontSize: 10,
            color: 'var(--text-3)',
            letterSpacing: '0.04em'
          }}>
            {modelLoaded ? `MODEL_LOADED · ${device.toUpperCase()}` : 'MODEL OFFLINE'}
          </span>
        </div>
      </div>

      {/* main layout */}
      <div style={{
        display: 'flex',
        height: 'calc(100vh - 49px)'
      }}>

        {/* chat — left */}
        <div style={{
          flex: 1,
          borderRight: '6px solid var(--border)',
          padding: 24,
          display: 'flex',
          flexDirection: 'column'
        }}>
          <ChatPanel
            messages={messages}
            onSend={handleSend}
            onReset={handleReset}
            loading={loading}
          />
        </div>

        {/* memory proof — right */}
        <div style={{
          width: 380,
          padding: 24,
          display: 'flex',
          flexDirection: 'column'
        }}>
          <MemoryProof
            result={compareResult}
            loading={compareLoading}
            onProve={handleProve}
          />
        </div>
      </div>
    </main>
  )
}