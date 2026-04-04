import { HealthStatus } from '@/types'

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export async function generate(prompt: string): Promise<string> {
  const res = await fetch(`${API_URL}/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt })
  })
  if (!res.ok) throw new Error('Generation failed')
  const data = await res.json()
  return data.response
}

export async function compare(prompt: string): Promise<{ttt_on: string, ttt_off: string}> {
  const res = await fetch(`${API_URL}/compare`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt })
  })
  if (!res.ok) throw new Error('Compare failed')
  const data = await res.json()
  return { ttt_on: data.ttt_on, ttt_off: data.ttt_off }
}

export async function reset(): Promise<void> {
  const res = await fetch(`${API_URL}/reset`, { method: 'POST' })
  if (!res.ok) throw new Error('Reset failed')
}

export async function health(): Promise<HealthStatus> {
  const res = await fetch(`${API_URL}/health`)
  if (!res.ok) throw new Error('Health check failed')
  return res.json()
}