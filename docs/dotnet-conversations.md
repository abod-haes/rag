# .NET conversation integration

The trusted .NET backend should call the RAG service. Mobile and web clients should
never receive `RAG_API_KEY` or call the RAG service directly.

## Recommended ownership

Use the .NET database as the application source of truth for users, permissions,
conversation lists, and UI messages. Store the RAG conversation ID beside the
application conversation so the RAG service can maintain retrieval context.

Suggested application table:

```text
ChatConversations
- Id                 application conversation GUID
- UserId             application user ID
- ProjectId          course/school/tenant ID
- RagConversationId  GUID returned by the RAG API
- Title
- CreatedAt
- UpdatedAt
```

The .NET backend may also store messages for UI history and auditing. The RAG
service stores its own copy only for AI follow-up context.

## Request flow

1. When the user creates a new chat, call `POST /api/chat/conversations`.
2. Store the returned `id` as `RagConversationId`.
3. Send every message to `POST /api/chat/ask` with that `conversationId`.
4. Do not send `documentIds` during normal chat. The RAG service selects the
   relevant document automatically.
5. If `needsClarification` is `true`, show `answer` to the user. Their next reply
   must use the same `conversationId`; a reply such as `رياضيات` is combined with
   the unresolved previous question.
6. When the application conversation is deleted, optionally call
   `DELETE /api/chat/conversations/{conversationId}`.

## Headers

```http
X-API-Key: server-side-rag-key
X-User-Id: authenticated-application-user-id
X-Project-Id: course-or-tenant-id
```

The same `X-User-Id` and `X-Project-Id` values must be sent for creating the
conversation, asking questions, loading messages, and deleting the conversation.

## Example HttpClient registration

```csharp
builder.Services.Configure<RagOptions>(
    builder.Configuration.GetSection("Rag"));

builder.Services.AddHttpClient<RagClient>((serviceProvider, client) =>
{
    var options = serviceProvider
        .GetRequiredService<IOptions<RagOptions>>()
        .Value;

    client.BaseAddress = new Uri(options.BaseUrl);
    client.Timeout = TimeSpan.FromMinutes(3);
});

public sealed class RagOptions
{
    public required string BaseUrl { get; init; }
    public required string ApiKey { get; init; }
}
```

## DTOs

```csharp
public sealed record CreateRagConversationRequest(string? Title);

public sealed record CreateRagConversationResponse(
    Guid Id,
    string Title,
    IReadOnlyList<Guid> ActiveDocumentIds);

public sealed record RagAskRequest(
    string Question,
    Guid ConversationId);

public sealed record RagDocumentSelection(
    Guid DocumentId,
    string Name,
    string FileName,
    string Subject,
    double Score,
    string SelectionReason);

public sealed record RagSource(
    string SourceId,
    Guid DocumentId,
    string Name,
    string FileName,
    int? PageNumber,
    int ChunkIndex,
    string? SectionTitle,
    string ContentType,
    bool IsNeighbor,
    double Score);

public sealed record RagAskResponse(
    Guid ConversationId,
    bool NeedsClarification,
    string RoutingStatus,
    string ResolvedQuestion,
    string? ClarificationQuestion,
    IReadOnlyList<string>? CandidateSubjects,
    IReadOnlyList<RagDocumentSelection>? CandidateDocuments,
    IReadOnlyList<RagDocumentSelection> SelectedDocuments,
    string Answer,
    IReadOnlyList<RagSource> Sources);
```

## RAG client

```csharp
using System.Net;
using System.Net.Http.Json;
using Microsoft.Extensions.Options;

public sealed class RagClient
{
    private readonly HttpClient _httpClient;
    private readonly RagOptions _options;

    public RagClient(HttpClient httpClient, IOptions<RagOptions> options)
    {
        _httpClient = httpClient;
        _options = options.Value;
    }

    public async Task<CreateRagConversationResponse> CreateConversationAsync(
        string userId,
        string projectId,
        string? title,
        CancellationToken cancellationToken = default)
    {
        using var request = CreateRequest(
            HttpMethod.Post,
            "/api/chat/conversations",
            userId,
            projectId);

        request.Content = JsonContent.Create(
            new CreateRagConversationRequest(title));

        using var response = await _httpClient.SendAsync(
            request,
            cancellationToken);

        await EnsureSuccessAsync(response, cancellationToken);
        return await response.Content
            .ReadFromJsonAsync<CreateRagConversationResponse>(cancellationToken)
            ?? throw new InvalidOperationException("RAG returned an empty response.");
    }

    public async Task<RagAskResponse> AskAsync(
        string userId,
        string projectId,
        Guid ragConversationId,
        string question,
        CancellationToken cancellationToken = default)
    {
        using var request = CreateRequest(
            HttpMethod.Post,
            "/api/chat/ask",
            userId,
            projectId);

        request.Content = JsonContent.Create(
            new RagAskRequest(question, ragConversationId));

        using var response = await _httpClient.SendAsync(
            request,
            cancellationToken);

        await EnsureSuccessAsync(response, cancellationToken);
        return await response.Content
            .ReadFromJsonAsync<RagAskResponse>(cancellationToken)
            ?? throw new InvalidOperationException("RAG returned an empty response.");
    }

    public async Task DeleteConversationAsync(
        string userId,
        string projectId,
        Guid ragConversationId,
        CancellationToken cancellationToken = default)
    {
        using var request = CreateRequest(
            HttpMethod.Delete,
            $"/api/chat/conversations/{ragConversationId}",
            userId,
            projectId);

        using var response = await _httpClient.SendAsync(
            request,
            cancellationToken);

        if (response.StatusCode == HttpStatusCode.NotFound)
            return;

        await EnsureSuccessAsync(response, cancellationToken);
    }

    private HttpRequestMessage CreateRequest(
        HttpMethod method,
        string path,
        string userId,
        string projectId)
    {
        var request = new HttpRequestMessage(method, path);
        request.Headers.Add("X-API-Key", _options.ApiKey);
        request.Headers.Add("X-User-Id", userId);
        request.Headers.Add("X-Project-Id", projectId);
        return request;
    }

    private static async Task EnsureSuccessAsync(
        HttpResponseMessage response,
        CancellationToken cancellationToken)
    {
        if (response.IsSuccessStatusCode)
            return;

        var body = await response.Content.ReadAsStringAsync(cancellationToken);
        throw new HttpRequestException(
            $"RAG request failed with {(int)response.StatusCode}: {body}");
    }
}
```

## Application service flow

```csharp
public async Task<RagAskResponse> SendMessageAsync(
    Guid applicationConversationId,
    string authenticatedUserId,
    string projectId,
    string text,
    CancellationToken cancellationToken)
{
    var conversation = await _db.ChatConversations
        .SingleAsync(
            item => item.Id == applicationConversationId
                && item.UserId == authenticatedUserId,
            cancellationToken);

    var ragResponse = await _ragClient.AskAsync(
        authenticatedUserId,
        projectId,
        conversation.RagConversationId,
        text,
        cancellationToken);

    _db.ChatMessages.Add(new ChatMessage
    {
        ConversationId = conversation.Id,
        Role = "user",
        Content = text,
    });

    _db.ChatMessages.Add(new ChatMessage
    {
        ConversationId = conversation.Id,
        Role = "assistant",
        Content = ragResponse.Answer,
        SourcesJson = JsonSerializer.Serialize(ragResponse.Sources),
    });

    conversation.UpdatedAt = DateTimeOffset.UtcNow;
    await _db.SaveChangesAsync(cancellationToken);

    return ragResponse;
}
```

When `NeedsClarification` is true, store and display the assistant response in the
same way. The next user message must continue with the same RAG conversation ID.
