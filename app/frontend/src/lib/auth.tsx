import { createContext, useContext, useEffect, useState, type ReactNode } from "react";
import { api, clearPassword, getPassword, setPassword } from "./api";

interface AuthState {
  ready: boolean;
  required: boolean;
  authed: boolean;
  login: (pw: string) => Promise<boolean>;
  logout: () => void;
}

const AuthContext = createContext<AuthState>({
  ready: false,
  required: false,
  authed: false,
  login: async () => false,
  logout: () => {},
});

export function AuthProvider({ children }: { children: ReactNode }) {
  const [ready, setReady] = useState(false);
  const [required, setRequired] = useState(false);
  const [authed, setAuthed] = useState(false);

  useEffect(() => {
    (async () => {
      try {
        const status = await api.authStatus();
        setRequired(status.required);
        if (!status.required) {
          setAuthed(true);
        } else {
          const pw = getPassword();
          if (pw) {
            try {
              await api.login(pw);
              setAuthed(true);
            } catch {
              clearPassword();
            }
          }
        }
      } catch {
        /* backend not reachable yet */
      } finally {
        setReady(true);
      }
    })();
  }, []);

  const login = async (pw: string) => {
    try {
      await api.login(pw);
      setPassword(pw);
      setAuthed(true);
      return true;
    } catch {
      return false;
    }
  };

  const logout = () => {
    clearPassword();
    setAuthed(false);
  };

  return (
    <AuthContext.Provider value={{ ready, required, authed, login, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export const useAuth = () => useContext(AuthContext);
