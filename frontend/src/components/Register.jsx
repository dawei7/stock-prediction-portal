import React, {useState} from 'react'
import axios from 'axios'
import {FontAwesomeIcon} from '@fortawesome/react-fontawesome'
import {faSpinner} from '@fortawesome/free-solid-svg-icons'

const Register = () => {
  const [username, setUsername] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [errors, setErrors] = useState({})
  const [success, setSuccess] = useState(false)
  const [loading, setLoading] = useState(false)

  const handleRegistration = async (e) => {
    e.preventDefault()
    setLoading(true)
    // check if password and confirm password match
    if(password !== confirmPassword){
      setErrors({confirmPassword: 'Passwords do not match'})
      return
    }
    const userData = {
      username,
      email,
      password
    }
    try{
        const response = await axios.post('http://localhost:8000/api/v1/register/', userData)
        console.log(response.data)
        console.log('User registered successfully')
        setErrors({})
        setSuccess(true)
      }catch(err){
        setErrors(err.response.data)
    }finally{
      setLoading(false)
    }
  }

  return (
    <>
     <div className='container'>
      <div className='row justify-content-center'>
        <div className='col-md-6 bg-light-dark p-5 rounded'>
          <h3 className='text-light text-center mb-4'>Create an Account</h3>
          <form onSubmit={handleRegistration}>
            <div className="mb-3">
              <input type="text" className='form-control' placeholder='Enter username' value={username} onChange={(e)=>setUsername(e.target.value)}/>
              <small>{errors.username && <div className="text-danger">{errors.username}</div>}</small>
            </div>
            <div className="mb-3">
              <input type="email" className='form-control' placeholder='Enter email' value={email} onChange={(e)=>setEmail(e.target.value)}/>
              <small>{errors.email && <div className="text-danger">{errors.email}</div>}</small>
            </div>
            <div className="mb-3">
              <input type="password" className='form-control' placeholder='Enter password' value={password} onChange={(e)=>setPassword(e.target.value)}/>
              <small>{errors.password && <div className="text-danger">{errors.password}</div>}</small>
            </div>
            <div className="mb-3">
              <input type="password" className='form-control' placeholder='Confirm password' value={confirmPassword} onChange={(e)=>setConfirmPassword(e.target.value)}/>
              <small>{errors.confirmPassword && <div className="text-danger">{errors.confirmPassword}</div>}</small>
            </div>
              {success && <div className="alert alert-success">User registered successfully</div>}
              {loading ?  (
                <button className='btn btn-info d-block mx-auto' type='button' disabled><FontAwesomeIcon icon={faSpinner} spin/>
                  &nbsp;Please wait...
                </button>
              ) : (
                <button type='submit' className='btn btn-info d-block mx-auto'>Register</button>
              )}
          </form>
        </div>
      </div>
     </div>
    </>
  )
}

export default Register