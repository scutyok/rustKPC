Name "cathpatrol1a"

AbortOnDamage TRUE
AbortOnEnemySighted FALSE
SightRange 512
TimeOut 999.0
FaceMoveDir TRUE
YawRate 180
FieldOfView 70
WalkSpeed 180
CheckForEnemyInterval 0.5
TargetPlayer


Loop1:

OnEnemyOutOfRange_Goto DISABLE		
OnTimerExpired_Goto DISABLE	
OnEnemySighted_Goto RunAway							
TimeOut 10
StartAnim			"AmbA"
Wait                            10
GoTo Loop1

RunAway:
					
StartAnim			"Run"
TurnTo				pt1a
MoveTo				pt1a	
TurnTo				pt2a
MoveTo				pt2a
TurnTo				pt1a	
StartAnim	"AmbA"
Wait		4
End



