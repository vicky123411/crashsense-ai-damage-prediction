using System;
using System.Collections;
using System.Text;
using UnityEngine;
using UnityEngine.Networking;

namespace CollisionDamage
{
    public enum PredictionEngine { Auto, Physics, Model }

    // Sends one collision to the local Python server and hands back the damage report.
    // Start the server first:  python -m crashdamage serve --model models/gnn_demo.pt
    public class PredictionClient : MonoBehaviour
    {
        public string serverUrl = "http://127.0.0.1:8765";
        public PredictionEngine engine = PredictionEngine.Auto;
        public int timeoutSeconds = 5;

        public IEnumerator Predict(CollisionRequest request, Action<DamageReport> onDone, Action<string> onError)
        {
            string json = JsonUtility.ToJson(request);
            string url = serverUrl.TrimEnd('/') + "/predict?engine=" + engine.ToString().ToLowerInvariant();

            using (UnityWebRequest www = new UnityWebRequest(url, UnityWebRequest.kHttpVerbPOST))
            {
                www.uploadHandler = new UploadHandlerRaw(Encoding.UTF8.GetBytes(json));
                www.downloadHandler = new DownloadHandlerBuffer();
                www.SetRequestHeader("Content-Type", "application/json");
                www.timeout = timeoutSeconds;

                yield return www.SendWebRequest();

                if (www.result != UnityWebRequest.Result.Success)
                {
                    if (onError != null) onError(www.error + " " + www.downloadHandler.text);
                    yield break;
                }

                DamageReport report = null;
                string parseError = null;
                try
                {
                    report = JsonUtility.FromJson<DamageReport>(www.downloadHandler.text);
                }
                catch (Exception e)
                {
                    parseError = e.Message;
                }

                if (parseError != null || report == null || report.parts == null)
                {
                    if (onError != null) onError("unexpected response: " + (parseError ?? www.downloadHandler.text));
                    yield break;
                }
                if (onDone != null) onDone(report);
            }
        }
    }
}
