import { useCallback } from "react";
import { create } from "zustand";

// Central app state so every figure KEEPS its settings/progress when you navigate away and back.
type Bag = Record<string, unknown>;
type Store = { bag: Bag; set: (k: string, v: unknown) => void };

export const useStore = create<Store>((set) => ({
  bag: {},
  set: (k, v) => set((s) => ({ bag: { ...s.bag, [k]: v } })),
}));

// A useState that persists in the global store under `key` (survives unmount / route changes).
export function usePersistentState<T>(key: string, initial: T): [T, (v: T | ((p: T) => T)) => void] {
  const value = useStore((s) => (key in s.bag ? (s.bag[key] as T) : initial));
  const setBag = useStore((s) => s.set);
  const setValue = useCallback(
    (v: T | ((p: T) => T)) => {
      const cur = key in useStore.getState().bag ? (useStore.getState().bag[key] as T) : initial;
      setBag(key, typeof v === "function" ? (v as (p: T) => T)(cur) : v);
    },
    [key] // eslint-disable-line
  );
  return [value, setValue];
}

// Non-reactive persistent ref (for heavy objects like trained models) — lives across navigation.
const refs = new Map<string, unknown>();
export function getFigureRef<T>(key: string, init: () => T): T {
  if (!refs.has(key)) refs.set(key, init());
  return refs.get(key) as T;
}
export function resetFigureRef<T>(key: string, init: () => T): T {
  refs.set(key, init());
  return refs.get(key) as T;
}
