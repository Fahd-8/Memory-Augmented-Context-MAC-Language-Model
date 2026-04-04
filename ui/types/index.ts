export interface Message {
    id: string
    role: 'user' | 'assistant'
    content: string
    timestamp: Date
  }
  
  export interface CompareResult {
    prompt: string
    ttt_on: string
    ttt_off: string
  }
  
  export interface HealthStatus {
    status: string
    model_loaded: boolean
    device: string
  }